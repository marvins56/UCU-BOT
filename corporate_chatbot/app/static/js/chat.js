// app/static/js/chat.js
class ChatInterface {
    constructor() {
        this.messageInput = document.querySelector('#message-input');
        this.chatContainer = document.querySelector('#chat-messages');
    }

    async sendMessage(message) {
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message })
            });
            
            const data = await response.json();
            this.displayMessage(message, 'user');
            this.displayMessage(data.response, 'bot');
        } catch (error) {
            console.error('Error:', error);
        }
    }

    displayMessage(message, type) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', type);
        messageElement.textContent = message;
        this.chatContainer.appendChild(messageElement);
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }
}