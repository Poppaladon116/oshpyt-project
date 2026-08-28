import * as fs from 'fs';
import { OshpytTokenizer } from './tokenizer';

const tokenizer = new OshpytTokenizer();
const data = fs.readFileSync('data.txt', 'utf-8');

console.log("Updating OSHPYT Dictionary...");
tokenizer.buildVocab(data);
console.log("Done! Check vocab.json");