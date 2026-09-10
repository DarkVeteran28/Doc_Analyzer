const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const getErrorMessage = async (response, fallback) => {
  try {
    const body = await response.json();
    return body.detail || fallback;
  } catch {
    return fallback;
  }
};

// Upload a PDF document to the backend and trigger RAG indexing
export const uploadDocument = async (file) => {
  const formData = new FormData();

  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/documents/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await getErrorMessage(response, "Failed to upload document"));
  }

  return response.json();
};

// Send a question about an uploaded document
export const askQuestion = async (question, documentId) => {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      document_id: documentId,
    }),
  });

  if (!response.ok) {
    throw new Error(await getErrorMessage(response, "Failed to get answer"));
  }

  return response.json();
};
