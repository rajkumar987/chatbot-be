import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import {
  HarmBlockThreshold,
  HarmCategory,
  TaskType,
} from "@google/generative-ai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { CSVLoader } from "langchain/document_loaders/fs/csv";
import { JSONLoader } from "langchain/document_loaders/fs/json";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { formatDocumentsAsString } from "langchain/util/document";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

export const createStore = async () => {
  const loader = new DirectoryLoader("./docs", {
    ".json": (path) => new JSONLoader(path),
    ".txt": (path) => new TextLoader(path),
    ".csv": (path) => new CSVLoader(path, { separator: "," }),
    ".pdf": (path) => new PDFLoader(path),
  });
  const docs = await loader.load();
  const csvContent = docs.map((doc) => doc.pageContent);

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 100,
  });

  const splitDocs = await textSplitter.createDocuments(csvContent);

  const vectorStore = await HNSWLib.fromDocuments(
    splitDocs,
    new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GOOGLE_API_KEY,
      modelName: "embedding-001",
      taskType: TaskType.RETRIEVAL_DOCUMENT,
    })
  );
  return vectorStore;
};

export const createChain = async (vectorStore) => {
  const model = new ChatGoogleGenerativeAI({
    apiKey: process.env.GOOGLE_API_KEY,
    modelName: "gemini-pro",
    maxOutputTokens: 2048,
    safetySettings: [
      {
        category: HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
      },
    ],
  });
  // You are an assistant bot .Your job is to make the customer feel heard and understand.
  // Reflect on the input you receive
  const questionPrompt = ChatPromptTemplate.fromTemplate(`
      You are an assistant bot dedicated to order-related inquiries.
    
      ----------------
      CONTEXT: {context}
      ----------------
      CUSTOMER INQUIRY: {question}
      ----------------
      CHAT HISTORY: {chatHistory}
      ----------------
      Response:
      `);

  // Helpful Answer:
  const retriever = vectorStore.asRetriever({
    k: 3,
  });

  const chain = RunnableSequence.from([
    {
      question: (input) => input.question,
      chatHistory: (input) => input.chatHistory ?? "",
      context: async (input) => {
        const relevantDocs = await retriever.getRelevantDocuments(
          input.question
        );
        const serialized = formatDocumentsAsString(relevantDocs);
        return serialized;
      },
    },
    questionPrompt,
    model,
    new StringOutputParser(),
  ]);
  return chain;
};
