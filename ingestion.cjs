const { MongoClient } = require('mongodb');
const { OpenAIEmbeddings } = require('@langchain/openai');
const { RecursiveCharacterTextSplitter } = require('@langchain/textsplitters');
const { htmlToText } = require('html-to-text');
const fs = require('fs');
const path = require('path');
const { parse } = require('csv-parse');

require('dotenv').config({ path: '.env.local' });

const csvFilePath = path.join(__dirname, 'supportdocs.csv');

function parseCSV(filePath) {
    return new Promise((resolve, reject) => {
        const records = [];
        fs.createReadStream(filePath)
            .pipe(parse({ columns: true, skip_empty_lines: true }))
            .on('data', (row) => {
                records.push(row);
            })
            .on('end', () => {
                resolve(records);
            })
            .on('error', (err) => {
                reject(err);
            });
    });
}

async function main() {
    try {
        const articles = await parseCSV(csvFilePath);
        console.log(`Parsed ${articles.length} articles.`);

        // Preprocess HTML to plain text
        const processedArticles = articles.map(article => ({
            ...article,
            PlainText: htmlToText(article.Content || '', {
                wordwrap: false,
                selectors: [
                    { selector: 'a', options: { ignoreHref: true } },
                    { selector: 'img', format: 'skip' }
                ]
            })
        }));

        
        // Chunk all articles
        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200
        });

        const chunkedArticles = [];
        for (const article of processedArticles) {
            const chunks = await splitter.splitText(article.PlainText);
            chunks.forEach((chunk, idx) => {
                chunkedArticles.push({
                    ...article,
                    chunk,
                    chunkIndex: idx
                });
            });
        }

        console.log(`Total chunks across all articles: ${chunkedArticles.length}`);

        // Test MongoDB connection before embedding
        const mongoUrl = process.env.MONGODB_URL;
        if (!mongoUrl) {
            throw new Error('MONGODB_URL not set in environment');
        }
        const client = new MongoClient(mongoUrl, { serverSelectionTimeoutMS: 5000 });
        try {
            await client.connect();
            await client.db('admin').command({ ping: 1 });
            console.log('MongoDB connection successful.');
        } catch (err) {
            console.error('MongoDB connection failed:', err);
            process.exit(1);
        } finally {
            await client.close();
        }

        // Proceed with embedding and inserting in batches
        const openaiApiKey = process.env.OPENAI_API_KEY;
        if (!openaiApiKey) {
            throw new Error('OPENAI_API_KEY not set in environment');
        }
        const embedder = new OpenAIEmbeddings({ openAIApiKey: openaiApiKey });

        // Connect to MongoDB for insertion
        console.log('connecting to mongodb');
        const client2 = new MongoClient(mongoUrl);
        await client2.connect();
        console.log('connected to mongodb');
        const db = client2.db('supportdocs');
        const collection = db.collection('embeddings');

        // Process in batches: embed then insert each batch
        const batchSize = 20;
        let totalInserted = 0;
        const totalBatches = Math.ceil(chunkedArticles.length / batchSize);
        console.log(`Processing ${chunkedArticles.length} chunks in ${totalBatches} batches of ${batchSize}...`);

        for (let i = 0; i < chunkedArticles.length; i += batchSize) {
            const batchNum = Math.floor(i / batchSize) + 1;
            const batch = chunkedArticles.slice(i, i + batchSize);
            
            // Embed the batch
            console.log(`Batch ${batchNum}/${totalBatches}: Embedding ${batch.length} chunks...`);
            const embeddedBatch = [];
            for (const chunkObj of batch) {
                const embedding = await embedder.embedQuery(chunkObj.chunk);
                embeddedBatch.push({
                    id: chunkObj.id,
                    title: chunkObj.Title,
                    date: chunkObj.Date,
                    permalink: chunkObj.Permalink,
                    categories: chunkObj.Categories,
                    chunk: chunkObj.chunk,
                    chunkIndex: chunkObj.chunkIndex,
                    embedding
                });
            }
            
            // Insert the batch
            const result = await collection.insertMany(embeddedBatch);
            totalInserted += result.insertedCount;
            console.log(`Batch ${batchNum}/${totalBatches}: Inserted ${result.insertedCount} documents (${totalInserted}/${chunkedArticles.length} total)`);
        }

        console.log(`Finished inserting ${totalInserted} documents into MongoDB.`);
        await client2.close();
    } catch (err) {
        console.error('Error parsing CSV:', err);
    }
}

main();