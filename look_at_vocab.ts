import * as fs from 'fs';

try {
    const data = JSON.parse(fs.readFileSync('vocab.json', 'utf8'));
    console.log("--- OSHPYT DICTIONARY CONTENT ---");
    // Look specifically for the commands
    const commands = Object.keys(data).filter(key => key.includes('cmd'));
    console.log("Recognized Commands:", commands);
    console.log("Total Tokens:", Object.keys(data).length);
} catch (e) {
    console.log("Error: vocab.json not found. You must run train.ts first.");
}