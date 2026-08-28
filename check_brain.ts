import { OshpytTokenizer } from './tokenizer';

const tokenizer = new OshpytTokenizer();
tokenizer.loadVocab();

const testInput = "[cmd:primarybutton]";
const encoded = tokenizer.encode(testInput);

console.log("--- TOKENIZER SANITY CHECK ---");
console.log(`Input String: "${testInput}"`);
console.log("Tokens Found:", encoded);

if (encoded.includes(0)) {
    console.log("RESULT: FAILED. The tokenizer found an [UNK] (0).");
    console.log("Check if your data.txt uses exactly [CMD:PrimaryButton].");
} else {
    console.log("RESULT: SUCCESS. The brain recognizes this command.");
}