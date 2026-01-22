const { MongoClient } = require('mongodb');
const { OpenAIEmbeddings } = require('@langchain/openai');

require('dotenv').config();

const openaiApiKey = process.env.OPENAI_API_KEY;
const mongoUrl = process.env.MONGODB_URL;

// Simple retrieval function: find top-k most similar chunks for a query
async function retrieveRelevantChunks(query, k = 3) {
    // Embed the query
    const embedder = new OpenAIEmbeddings({ openAIApiKey: openaiApiKey });
    const queryEmbedding = await embedder.embedQuery(query);

    // Connect to MongoDB
    const client = new MongoClient(mongoUrl);
    await client.connect();
    const db = client.db('supportdocs');
    const collection = db.collection('embeddings');

    // Fetch all embeddings (for demo; for large sets, use $vectorSearch or similar in production)
    const docs = await collection.find({}).toArray();

    // Compute cosine similarity
    function cosineSim(a, b) {
        let dot = 0, normA = 0, normB = 0;
        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    const scored = docs.map(doc => ({
        ...doc,
        score: cosineSim(queryEmbedding, doc.embedding)
    }));
    scored.sort((a, b) => b.score - a.score);
    await client.close();
    return scored.slice(0, k);
}

async function main() {
    // --- Simple retrieval test ---
    const testQuery = 'What features were released in february 2021?';
    const topChunks = await retrieveRelevantChunks(testQuery, 3);
    console.log('Top retrieved chunk:', topChunks[0]?.chunk);
    console.log('Score:', topChunks[0]?.score);
    // --- End retrieval test ---
}

main();