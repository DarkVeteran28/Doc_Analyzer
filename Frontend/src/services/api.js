const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

// Upload a PDF document to the backend
export const uploadDocument = async (file) => {
  const formData = new FormData();

  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Failed to upload document");
  }

  return response.json();
};

// Send a question about an uploaded document
export const askQuestion = async (question, documentId) => {
  const response = await fetch(`${API_BASE_URL}/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      documentId,
    }),
  });

  if (!response.ok) {
    throw new Error("Failed to get answer");
  }

  return response.json();
};