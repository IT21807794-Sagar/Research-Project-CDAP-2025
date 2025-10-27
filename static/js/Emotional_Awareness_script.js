document.addEventListener('DOMContentLoaded', function() {
    // Sidebar toggle functionality (if sidebar exists on the page)
    const sidebar = document.getElementById('sidebar');
    const toggleBtn = document.querySelector('.toggle-btn');

    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener('click', function() {
            sidebar.classList.toggle('collapsed');
        });
    }

    // --- Chatbot setup ---
    const chatbotToggle = document.getElementById("chatbot-toggle");
    const chatbotPopup = document.getElementById("chatbot-popup");
    const chatbotClose = document.getElementById("chatbot-close");
    const sendBtn = document.getElementById("send-btn");
    const userInput = document.getElementById("user-message");
    const chatBox = document.getElementById("chat-box");
    const voiceBtn = document.getElementById("voice-btn");
    const expandBtn = document.getElementById("expand-btn");
    let recognition;
    let isExpanded = false;

    // --- STRESS-BASED AUTO-POPUP FUNCTIONALITY FOR ALL PAGES ---
    function initializeAutoPopup() {
        // Check initial stress level and open chatbot if needed
        const stressDiv = document.getElementById('stress-level');
        if (stressDiv) {
            const level = stressDiv.dataset.level ? stressDiv.dataset.level.toLowerCase() : '';
            if (level === "moderate" || level === "high") {
                openChatbotWithStressAlert(level);
            }
        }

        // Initialize stress monitoring for real-time updates (if stress element exists)
        if (stressDiv) {
            initializeStressMonitoring();
        }

        // Additional triggers for other pages (non-stress based)
        initializePageSpecificTriggers();
    }

    function initializeStressMonitoring() {
        // Stress level update function
        function updateStressLevel() {
            fetch('/get_stress_level')
                .then(res => res.json())
                .then(data => {
                    const stressDiv = document.getElementById('stress-level');
                    if (stressDiv) {
                        const oldLevel = stressDiv.dataset.level;
                        const newLevel = data.stress_level.toLowerCase();

                        // Update div only if value changed
                        if (oldLevel !== newLevel) {
                            stressDiv.dataset.level = newLevel;
                            stressDiv.textContent = `Stress Level: ${data.stress_level}`;

                            // Auto-open chatbot if moderate/high and chatbot is not already open
                            if ((newLevel === "moderate" || newLevel === "high") && chatbotPopup.style.display !== "flex") {
                                openChatbotWithStressAlert(newLevel);
                            }

                            // Update message if chatbot is already open
                            else if ((newLevel === "moderate" || newLevel === "high") && chatbotPopup.style.display === "flex") {
                                addStressUpdateMessage(newLevel);
                            }

                            // Auto-close chatbot if stress returns to low (optional)
                            if (newLevel === "low" && chatbotPopup.style.display === "flex") {
                                addStressRelievedMessage();
                            }
                        }
                    }
                })
                .catch(err => console.error("Error fetching stress level:", err));
        }

        // Update stress level every 5 seconds
        setInterval(updateStressLevel, 5000);
    }

    function initializePageSpecificTriggers() {
        const currentPage = getCurrentPage();

        switch(currentPage) {
            case 'group_chat':
                // Monitor for user inactivity or confusion in group chat
                setTimeout(() => {
                    if (!hasUserInteracted() && chatbotPopup.style.display !== "flex") {
                        openChatbotWithPageContext(currentPage);
                    }
                }, 45000); // 45 seconds
                break;

            case 'personalized_revision':
                // Check if user is stuck on a difficult question
                monitorRevisionProgress();
                break;

            case 'cognitive_assessment':
                // Popup after first few questions to offer help
                setTimeout(() => {
                    if (chatbotPopup.style.display !== "flex") {
                        openChatbotWithPageContext(currentPage);
                    }
                }, 30000); // 30 seconds
                break;

            case 'emotional_awareness':
                // Already handled by stress monitoring
                break;

            default:
                // General help after 60 seconds on any other page
                setTimeout(() => {
                    if (!hasUserInteracted() && chatbotPopup.style.display !== "flex") {
                        openChatbotWithPageContext('general');
                    }
                }, 60000);
        }
    }

    function openChatbotWithStressAlert(stressLevel) {
        chatbotPopup.style.display = "flex";
        if (chatbotToggle) chatbotToggle.style.display = "none";

        // Add stress-specific welcome message
        const welcomeMsg = document.createElement("div");
        welcomeMsg.classList.add("bot-msg");

        if (stressLevel === "high") {
            welcomeMsg.innerHTML = ``;
        } else {
            welcomeMsg.innerHTML = ``;
        }

        chatBox.appendChild(welcomeMsg);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function openChatbotWithPageContext(page) {
        chatbotPopup.style.display = "flex";
        if (chatbotToggle) chatbotToggle.style.display = "none";

        let welcomeMessage = '';

        switch(page) {
            case 'group_chat':
                welcomeMessage = "";
                break;
            case 'personalized_revision':
                welcomeMessage = "";
                break;
            case 'cognitive_assessment':
                welcomeMessage = "";
                break;
            case 'general':
                welcomeMessage = "";
                break;
            default:
                welcomeMessage = "";
        }

        const welcomeMsg = document.createElement("div");
        welcomeMsg.classList.add("bot-msg");
        welcomeMsg.innerHTML = `<p>${welcomeMessage}</p>`;
        chatBox.appendChild(welcomeMsg);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function addStressUpdateMessage(newLevel) {
        const updateMsg = document.createElement("div");
        updateMsg.classList.add("bot-msg");

        if (newLevel === "high") {
            updateMsg.innerHTML = ``;
        } else {
            updateMsg.innerHTML = ``;
        }

        chatBox.appendChild(updateMsg);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function addStressRelievedMessage() {
        const reliefMsg = document.createElement("div");
        reliefMsg.classList.add("bot-msg");
        reliefMsg.innerHTML = ``;
        chatBox.appendChild(reliefMsg);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Helper functions
    function getCurrentPage() {
        const currentUrl = window.location.pathname;
        if (currentUrl.includes('GC') || currentUrl.includes('group_chat') || currentUrl.includes('group')) return 'group_chat';
        if (currentUrl.includes('PR') || currentUrl.includes('personalized_revision') || currentUrl.includes('revision')) return 'personalized_revision';
        if (currentUrl.includes('CA') || currentUrl.includes('cognitive_assessment') || currentUrl.includes('assessment')) return 'cognitive_assessment';
        if (currentUrl.includes('Emotional_Awareness') || currentUrl.includes('video') || currentUrl.includes('stress')) return 'emotional_awareness';
        if (currentUrl.includes('home') || currentUrl.includes('index') || currentUrl === '/') return 'home';
        return 'general';
    }

    function hasUserInteracted() {
        // Check if user has clicked, typed, or scrolled recently
        // This is a simple implementation - you might want to track actual interactions
        return sessionStorage.getItem('userInteracted') === 'true';
    }

    function monitorRevisionProgress() {
        // Monitor for signs of struggle in revision (repeated wrong answers, long pauses, etc.)
        // This would need to be integrated with your actual revision system
        const wrongAnswers = document.querySelectorAll('.incorrect, .error, .wrong');
        if (wrongAnswers.length >= 3) {
            openChatbotWithPageContext('personalized_revision');
        }
    }

    // Track user interactions
    document.addEventListener('click', () => sessionStorage.setItem('userInteracted', 'true'));
    document.addEventListener('keypress', () => sessionStorage.setItem('userInteracted', 'true'));

    // Initialize auto-popup for all pages
    initializeAutoPopup();

    // --- Toggle chatbot popup ---
    if (chatbotToggle) {
        chatbotToggle.addEventListener("click", () => {
            chatbotPopup.style.display = "flex";
            chatbotToggle.style.display = "none";
        });
    }

    if (chatbotClose) {
        chatbotClose.addEventListener("click", () => {
            if (isExpanded) {
                toggleExpand();
            } else {
                chatbotPopup.style.display = "none";
                if (chatbotToggle) chatbotToggle.style.display = "block";
            }
        });
    }

    // --- Toggle expand/collapse ---
    if (expandBtn) {
        expandBtn.addEventListener("click", toggleExpand);
    }

    // --- Speech-to-text for user input ---
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.lang = "en-US";
        recognition.interimResults = false;
        recognition.continuous = false;

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            if (userInput) userInput.value = transcript;
        };

        recognition.onerror = (event) => {
            console.error("Speech recognition error:", event.error);
        };
    }

    if (voiceBtn) {
        voiceBtn.addEventListener("click", () => {
            if (recognition) recognition.start();
        });
    }

    // --- Text-to-speech function ---
    function speakMessage(message) {
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(message);
            utterance.lang = "en-US";
            speechSynthesis.speak(utterance);
        }
    }

    // --- Send user message ---
    if (sendBtn && userInput) {
        sendBtn.addEventListener("click", sendMessage);

        // --- Press to send ---
        userInput.addEventListener("keypress", (e) => {
            if (e.key === "Enter") sendMessage();
        });
    }

    function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // --- Show user message ---
        const userDiv = document.createElement("div");
        userDiv.classList.add("user-msg");
        userDiv.textContent = `You: ${message}`;
        chatBox.appendChild(userDiv);
        userInput.value = '';

        // --- Fetch bot response ---
        fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: message,
                component: getCurrentPage(),
                stress_level: document.getElementById('stress-level')?.dataset.level || 'unknown'
            })
        })
        .then(res => res.json())
        .then(data => {
            const botMsg = data.response;

            // --- Create bot message div ---
            const botDiv = document.createElement("div");
            botDiv.classList.add("bot-msg");

            const botText = document.createElement("span");
            botText.classList.add("bot-text");
            botText.textContent = `Assistant: ${botMsg}`;

            const speakBtn = document.createElement("button");
            speakBtn.classList.add("speak-icon");
            speakBtn.textContent = "🔊";
            speakBtn.style.cursor = "pointer";
            speakBtn.style.marginLeft = "8px";
            speakBtn.style.border = "none";
            speakBtn.style.background = "none";
            speakBtn.style.fontSize = "16px";

            // Speak on click
            speakBtn.addEventListener("click", () => { speakMessage(botMsg); });

            // Append to chat box
            botDiv.appendChild(botText);
            botDiv.appendChild(speakBtn);
            chatBox.appendChild(botDiv);

            chatBox.scrollTop = chatBox.scrollHeight;
        })
        .catch(err => {
            console.error("Chat error:", err);
            // Fallback response
            const botDiv = document.createElement("div");
            botDiv.classList.add("bot-msg");
            botDiv.textContent = `Assistant: I'm having trouble connecting right now. How can I help you?`;
            chatBox.appendChild(botDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        });
    }

    // Toggle expand function
    function toggleExpand() {
        isExpanded = !isExpanded;

        if (isExpanded) {
            chatbotPopup.classList.add('expanded');
            expandBtn.innerHTML = '<i class="fas fa-compress"></i>';
            expandBtn.title = "Minimize chat";
        } else {
            chatbotPopup.classList.remove('expanded');
            expandBtn.innerHTML = '<i class="fas fa-expand"></i>';
            expandBtn.title = "Expand chat";
        }
    }

    // Auto-focus on input when chatbot opens
    if (chatbotToggle && userInput) {
        chatbotToggle.addEventListener("click", function() {
            setTimeout(() => {
                userInput.focus();
            }, 100);
        });
    }
});