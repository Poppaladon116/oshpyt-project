import crypto from "node:crypto";
import argon2 from "argon2";
import type { Request, Response } from "express";

type User = {
  id: string;
  email: string;
  passwordHash: string;
};

const usersByEmail = new Map<string, User>();

declare module "express-session" {
  interface SessionData {
    userId?: string;
    csrfToken?: string;
  }
}

function normalizeEmail(value: unknown): string | null {
  if (typeof value !== "string") return null;

  const email = value.trim().toLowerCase();

  if (email.length < 3 || email.length > 254) return null;
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return null;

  return email;
}

function getPassword(value: unknown): string | null {
  if (typeof value !== "string") return null;
  if (value.length < 12 || value.length > 1024) return null;
  return value;
}

function csrfToken(req: Request): string {
  if (!req.session.csrfToken) {
    req.session.csrfToken = crypto.randomBytes(32).toString("base64url");
  }

  return req.session.csrfToken;
}

function validCsrf(req: Request): boolean {
  const supplied = req.body?.csrfToken;

  return (
    typeof supplied === "string" &&
    typeof req.session.csrfToken === "string" &&
    supplied.length === req.session.csrfToken.length &&
    crypto.timingSafeEqual(
      Buffer.from(supplied),
      Buffer.from(req.session.csrfToken)
    )
  );
}

export function renderLoginForm(req: Request, res: Response): void {
  const token = csrfToken(req);

  res.type("html").send(
    `<form method="post" action="/auth/login" style="width:280px;padding:18px;box-sizing:border-box;border:1px solid #e5e7eb;border-radius:12px;background:#fff;font-family:Arial,sans-serif;">
      <input type="hidden" name="csrfToken" value="${token}">
      <label for="login-email" style="display:block;margin-bottom:6px;color:#374151;">email</label>
      <input id="login-email" name="email" type="email" autocomplete="email" required style="box-sizing:border-box;width:100%;padding:10px;border:1px solid #d1d5db;border-radius:8px;">
      <label for="login-password" style="display:block;margin:12px 0 6px;color:#374151;">password</label>
      <input id="login-password" name="password" type="password" autocomplete="current-password" required style="box-sizing:border-box;width:100%;padding:10px;border:1px solid #d1d5db;border-radius:8px;">
      <button type="submit" style="margin-top:14px;padding:10px 16px;border:0;border-radius:8px;background:#16a34a;color:#fff;font-weight:700;cursor:pointer;">sign in</button>
    </form>`
  );
}

export async function register(
  req: Request,
  res: Response
): Promise<void> {
  if (!validCsrf(req)) {
    res.status(403).json({ error: "Invalid CSRF token." });
    return;
  }

  const email = normalizeEmail(req.body?.email);
  const password = getPassword(req.body?.password);

  if (!email || !password) {
    res.status(400).json({ error: "Invalid registration data." });
    return;
  }

  if (usersByEmail.has(email)) {
    res.status(409).json({ error: "Unable to create account." });
    return;
  }

  const passwordHash = await argon2.hash(password, {
    type: argon2.argon2id,
    memoryCost: 19456,
    timeCost: 2,
    parallelism: 1
  });

  const user: User = {
    id: crypto.randomUUID(),
    email,
    passwordHash
  };

  usersByEmail.set(email, user);

  req.session.regenerate((error) => {
    if (error) {
      res.status(500).json({ error: "Could not create session." });
      return;
    }

    req.session.userId = user.id;
    req.session.csrfToken = crypto.randomBytes(32).toString("base64url");
    res.status(201).json({ ok: true });
  });
}

export async function login(req: Request, res: Response): Promise<void> {
  if (!validCsrf(req)) {
    res.status(403).json({ error: "Invalid CSRF token." });
    return;
  }

  const email = normalizeEmail(req.body?.email);
  const password = getPassword(req.body?.password);
  const user = email ? usersByEmail.get(email) : undefined;

  const verified =
    user && password
      ? await argon2.verify(user.passwordHash, password)
      : false;

  if (!verified || !user) {
    res.status(401).json({ error: "Invalid email or password." });
    return;
  }

  req.session.regenerate((error) => {
    if (error) {
      res.status(500).json({ error: "Could not create session." });
      return;
    }

    req.session.userId = user.id;
    req.session.csrfToken = crypto.randomBytes(32).toString("base64url");
    res.status(200).json({ ok: true });
  });
}

export function logout(req: Request, res: Response): void {
  if (!validCsrf(req)) {
    res.status(403).json({ error: "Invalid CSRF token." });
    return;
  }

  req.session.destroy(() => {
    res.clearCookie("oshpyt.sid");
    res.status(204).end();
  });
}