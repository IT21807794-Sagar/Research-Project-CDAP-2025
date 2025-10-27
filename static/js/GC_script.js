    const socket = io();
    let currentRoom = null;
    let currentUsername = null;
    let currentGroupId = null;
    let onlineMembers = new Set();

    function promptUsername(groupId) {
        const username = prompt("Enter your username to join the group chat:");
        if(username && username.trim() !== "") {
            openChat(groupId, username.trim());
        } else if (username !== null) {
            alert("Username cannot be empty!");
        }
    }

    function openChat(groupId, username) {
        document.getElementById("messages").innerHTML = "";
        document.getElementById("welcomeMessage").style.display = "block";

        currentGroupId = groupId;
        currentRoom = "group" + groupId;
        currentUsername = username;
        onlineMembers.clear();

        document.getElementById("groupNumber").textContent = "#" + (groupId + 1);
        document.getElementById("memberCount").textContent = "1";
        document.getElementById("chatModal").style.display = "flex";

        socket.emit('join', {username: currentUsername, room: currentRoom});
        loadHistory(currentRoom);
        
        // Add user to online members
        onlineMembers.add(currentUsername);
        updateMemberCount();
    }

    function closeChat() {
        document.getElementById("chatModal").style.display = "none";
        if(currentRoom && currentUsername){
            socket.emit('leave', {username: currentUsername, room: currentRoom});
            onlineMembers.delete(currentUsername);
            currentRoom = null;
            currentUsername = null;
            currentGroupId = null;
        }
    }

    function sendMessage() {
        const input = document.getElementById("msg-input");
        const msg = input.value.trim();
        if(msg !== '' && currentRoom) {
            const data = {
                room: currentRoom, 
                username: currentUsername, 
                msg: msg, 
                timestamp: new Date().toLocaleTimeString()
            };
            socket.emit('text', data);
            saveMessage(currentRoom, data);
            input.value = '';
            document.getElementById("welcomeMessage").style.display = "none";
        }
    }

    document.getElementById("msg-input").addEventListener("keypress", function(e) {
        if(e.key === "Enter") sendMessage();
    });

    function saveMessage(room, data){
        const messages = JSON.parse(localStorage.getItem(room) || '[]');
        messages.push(data);
        if(messages.length > 50) messages.shift();
        localStorage.setItem(room, JSON.stringify(messages));
    }

    function loadHistory(room) {
        const messages = JSON.parse(localStorage.getItem(room) || '[]');
        const ul = document.getElementById("messages");
        ul.innerHTML = "";

        if (messages.length > 0) {
            document.getElementById("welcomeMessage").style.display = "none";
        }

        messages.forEach(data => renderMessage(data));
    }

    function renderMessage(data) {
        const ul = document.getElementById("messages");
        const li = document.createElement("li");
        
        if(data.username && data.username === currentUsername){
            li.className = "my-message";
            li.innerHTML = `
                <div class="message-bubble">
                    <div class="message-content">${data.msg}</div>
                    <div class="message-time">${data.timestamp || ''}</div>
                </div>
                <div class="message-avatar">${getUserIcon(data.username)}</div>
            `;
        } else if(data.username){
            li.className = "other-message";
            li.innerHTML = `
                <div class="message-avatar">${getUserIcon(data.username)}</div>
                <div class="message-bubble">
                    <div class="sender">${data.username}</div>
                    <div class="message-content">${data.msg}</div>
                    <div class="message-time">${data.timestamp || ''}</div>
                </div>
            `;
        } else {
            li.className = "status-message";
            li.innerHTML = `<div class="status-content">${data.msg}</div>`;
        }
        ul.appendChild(li);
        ul.scrollTop = ul.scrollHeight;
    }

    function getUserIcon(username) {
        const icons = ['👤', '🦸', '👨‍🎓', '👩‍💻', '🧑‍🏫', '👨‍🔬', '👩‍🎨', '🧑‍🚀'];
        const index = username.length % icons.length;
        return icons[index];
    }

    function updateMemberCount() {
        document.getElementById("memberCount").textContent = onlineMembers.size;
    }

    // Socket event handlers
    socket.on('message', function(data){
        if(!data.msg) return;
        data.timestamp = new Date().toLocaleTimeString();
        saveMessage(data.room, data);
        document.getElementById("welcomeMessage").style.display = "none";
        renderMessage(data);
    });

    socket.on('status', function(data){
        if(!data.msg) return;
        const ul = document.getElementById("messages");
        const li = document.createElement("li");
        li.className = "status-message";
        li.innerHTML = `<div class="status-content">${data.msg}</div>`;
        ul.appendChild(li);
        ul.scrollTop = ul.scrollHeight;
        
        // Update online members
        if(data.msg.includes('joined')) {
            onlineMembers.add(data.username);
        } else if(data.msg.includes('left')) {
            onlineMembers.delete(data.username);
        }
        updateMemberCount();
    });

    // Close modal when clicking outside
    document.addEventListener('click', function(e) {
        if(e.target.classList.contains('modal')) {
            closeChat();
        }
    });

    // Close modal with Escape key
    document.addEventListener('keydown', function(e) {
        if(e.key === 'Escape' && document.getElementById('chatModal').style.display === 'flex') {
            closeChat();
        }
    });