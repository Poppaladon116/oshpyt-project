import * as fs from 'fs';

export class OshpytTokenizer {
    // We use Maps for O(1) lookup speed during deep inference
    public vocab: Map<string, number> = new Map();
    public inverseVocab: Map<number, string> = new Map();
    private nextId = 1;

    // THE GOD MODE REGEX: 
    // Catches long triggers, hex codes, frames, and symbols as individual units
    private readonly regex = /OSHPYT_TRIGGER_\w+|COMPONENT_END|FRAME_\w+|#?[a-f0-9]{6}|[\[\]:<>=";/]|[\w']+|[,.!?;]/gi;

    constructor() {
        this.addToken("[UNK]", 0);
    }

    addToken(word: string, id?: number) {
        const clean = word.toLowerCase().trim();
        if (clean === "" || this.vocab.has(clean)) return;
        
        const newId = id ?? this.nextId++;
        this.vocab.set(clean, newId);
        this.inverseVocab.set(newId, clean);
        
        // Ensure nextId stays ahead of loaded IDs
        if (newId >= this.nextId) this.nextId = newId + 1;
    }

    buildVocab(text: string) {
        console.log("OSHPYT: Analyzing textbook for unique concepts...");
        const matches = text.match(this.regex) || [];
        matches.forEach(w => this.addToken(w));
        this.saveVocab();
        console.log(`OSHPYT: Dictionary built with ${this.vocab.size} tokens.`);
    }

    encode(text: string): number[] {
        const matches = text.match(this.regex) || [];
        return matches
            .map(w => this.vocab.get(w.toLowerCase().trim()) ?? 0)
            .filter(id => id !== 0); // Discard UNK for cleaner logic
    }

    decode(ids: number[]): string {
        return ids.map(id => this.inverseVocab.get(id) || "").join(" ");
    }

    saveVocab() {
        // Convert Map to Object for JSON storage
        const obj = Object.fromEntries(this.vocab);
        fs.writeFileSync('vocab.json', JSON.stringify(obj, null, 2));
    }
    
    loadVocab() {
        if (fs.existsSync('vocab.json')) {
            const data = JSON.parse(fs.readFileSync('vocab.json', 'utf8'));
            this.vocab.clear();
            this.inverseVocab.clear();
            
            // This is the part you provided - it is correct!
            for (const [word, id] of Object.entries(data)) {
                this.vocab.set(word, id as number);
                this.inverseVocab.set(id as number, word);
            }
            
            // Sync nextId so new tokens don't overlap
            const ids = Array.from(this.vocab.values());
            this.nextId = Math.max(...ids) + 1;
            console.log(`OSHPYT: Dictionary loaded. ${this.vocab.size} tokens active.`);
        }
    }
}