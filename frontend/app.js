const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");
const chatBox = document.getElementById("chat-box");

sendButton.addEventListener("click", async () => {
    const question = userInput.value.trim();
    if (!question) {
    return;
    }
    chatBox.textContent = "Thinking...";
    sendButton.disabled = true; 
    


    try {
        const response = await fetch(
            "https://cs-tutor-api-32775674218.europe-west2.run.app/generate",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    prompt: question
                })
            }
        );
        if (!response.ok) {
            throw new Error(`Request failed with status ${response.status}`);
        }
        const data = await response.json();
        chatBox.textContent = data.response;
        userInput.value = "";
    } catch (error) {
        chatBox.textContent = "Something went wrong. Please try again.";
    } finally {
        sendButton.disabled = false; 
    }

});