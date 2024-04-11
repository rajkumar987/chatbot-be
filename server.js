import express from "express";
import cors from "cors";
import { createStore, createChain } from "./utils/index.js";
import { config } from "dotenv";
config();

const app = express();
app.use(express.json());

app.use(cors());

const PORT = process.env.PORT || 4000;

app.post("/api/chat", async (req, res) => {
  try {
    const { input_question, history } = req.body;

    if (!input_question || !Array.isArray(history) || history.length === 0) {
      return res.status(400).json({
        message: "Missing input_question or history in the request body",
      });
    }

    const vectorStore = await createStore();
    const chain = await createChain(vectorStore);
    const response = await chain.invoke({
      question: input_question,
      chat_history: history?.map((h) => h.content).join("\n"),
    });
    if (response.trim() === "") {
      response =
        "There was an error processing your request. Please try again after some time.";
    }
    return res.json({
      role: "assistant",
      content: response,
    });
  } catch (error) {
    return res.status(500).json({
      message: "Something went wrong",
      error: error.message,
    });
  }
});

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
