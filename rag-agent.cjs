const { MongoClient } = require('mongodb');
const { OpenAIEmbeddings, ChatOpenAI } = require('@langchain/openai');

require('dotenv').config({ path: '.env.local' });

const cliArgs = parseArgs();

const openaiApiKey = process.env.OPENAI_API_KEY;
const mongoUrl = process.env.MONGODB_URL;

// Parse CLI arguments
function parseArgs() {
	const args = process.argv.slice(2);
	const result = {};
	
	for (let i = 0; i < args.length; i++) {
		const arg = args[i];
		if (arg === '-query' || arg === '--query') {
			if (i + 1 < args.length && !args[i + 1].startsWith('-')) {
				result.query = args[i + 1];
				i++;
			} else {
				console.error('Error: -query/--query requires a value');
				console.error('Usage: node ./rag-agent.cjs -query "your question here" [-eval]');
				process.exit(1);
			}
		} else if (arg === '-eval' || arg === '--eval') {
			result.eval = true;
		}
	}
	
	return result;
}


// Retrieve top-k relevant chunks from MongoDB
async function retrieveRelevantChunks(query, k = 5) {
	const embedder = new OpenAIEmbeddings({ openAIApiKey: openaiApiKey });
	const queryEmbedding = await embedder.embedQuery(query);
	const client = new MongoClient(mongoUrl);
	await client.connect();
	const db = client.db('supportdocs');
	const collection = db.collection('embeddings');

    const results = await collection.aggregate([
        {
            $vectorSearch: {
                index: "supportdocs_index",
                path: "embedding",
                queryVector: queryEmbedding,
                numCandidates: 50,
                limit: k
            }
        }
    ]).toArray();
    
    await client.close();
    return results;
}

async function synthesizeAnswer(query, contextChunks) {
	const llm = new ChatOpenAI({ openAIApiKey: openaiApiKey, modelName: 'gpt-3.5-turbo', temperature: 0 });
	const contextText = contextChunks.map((c, i) => `Context ${i+1}:\n${c.chunk}`).join('\n\n');
	const prompt = `You are a helpful support assistant. Use the following context to answer the user's question.\n\n${contextText}\n\nQuestion: ${query}\nAnswer:`;
	const response = await llm.invoke(prompt);
	return response.content;
}

async function evaluateAnswerGroundedness(question, answer, contextChunks) {
    const llm = new ChatOpenAI({ openAIApiKey: openaiApiKey, model: 'gpt-4', temperature: 0 });
    const contextText = contextChunks.map((c, i) => `Context ${i+1}:\n${c.chunk}`).join('\n\n');
    
    const groundednessPrompt = `
    You are a groundedness evaluator. Your task is to determine if the given answer is fully supported by the provided context.

    An answer is GROUNDED if:
    - Every claim and statement in the answer can be traced back to information in the context
    - The answer does not include information that isn't present in the context
    - The answer does not make assumptions or inferences beyond what the context supports

    An answer is NOT GROUNDED if:
    - It contains claims not supported by the context (hallucinations)
    - It adds information or details not present in the context
    - It makes unsupported generalizations or conclusions

    Context:
    ${contextText}

    Question: ${question}

    Answer to evaluate:
    ${answer}

    Evaluate the groundedness of this answer. First, briefly explain your reasoning (2-3 sentences), then provide your verdict.

    Respond in this exact format:
    REASONING: <your brief explanation>
    VERDICT: <GROUNDED or NOT_GROUNDED>
    `;

    const response = await llm.invoke(groundednessPrompt);
    console.log('groundedness eval response');
    console.log(response.content);
}

async function evaluateAnswerPrecision(question, contextChunks) {
    const llm = new ChatOpenAI({ openAIApiKey: openaiApiKey, model: 'gpt-4', temperature: 0 });
    const contextText = contextChunks.map((c, i) => `Context ${i+1}:\n${c.chunk}`).join('\n\n');

    const precisionJudgePrompt = `
    You are a relevance judge. Your task is to determine if the retrieved document is relevant to answering the given question.

    A document is RELEVANT if it contains information that would help answer the question, even if it doesn't fully answer it.
    A document is NOT RELEVANT if it contains no useful information for answering the question.

    Question: ${question}

    Retrieved Document: ${contextText}

    Is this document relevant to answering the question?
    First, briefly explain your reasoning (2-3 sentences), then provide your verdict.

    Respond in this exact format:
    REASONING: <your brief explanation>
    VERDICT: <RELEVANT or NOT_RELEVANT>
    `;

    const response = await llm.invoke(precisionJudgePrompt);
    console.log('precision eval response:');
    console.log(response.content);
}

async function main() {
	const defaultQuery = 'what features were released in february?';
	const userQuery = cliArgs.query || defaultQuery;
	const topChunks = await retrieveRelevantChunks(userQuery, 5);
	console.log('Retrieved context chunks. Synthesizing answer...');
	const answer = await synthesizeAnswer(userQuery, topChunks);
	console.log('---\nUser question:', userQuery);
	console.log('Answer:', answer);

	if (cliArgs.eval) {
        console.log('context chunks used in answer');
        const contextText = topChunks.map((c, i) => `Context ${i+1}:\n${c.chunk}`).join('\n\n');
        console.log({contextText});
		console.log('evaluating for groundedness and precision...');
		await evaluateAnswerGroundedness(userQuery, answer, topChunks);
		await evaluateAnswerPrecision(userQuery, topChunks);
	}
}

main();
