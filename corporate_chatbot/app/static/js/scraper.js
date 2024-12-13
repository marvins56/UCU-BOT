document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('scraper-form');
    const resultsPanel = document.getElementById('results-panel');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const scrapedUrls = document.getElementById('scraped-urls');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const urls = document.getElementById('urls').value.split('\n').filter(url => url.trim());
        if (urls.length === 0) {
            alert('Please enter at least one URL');
            return;
        }

        // Show results panel
        resultsPanel.style.display = 'block';
        progressBar.style.width = '0%';
        scrapedUrls.innerHTML = '';

        const data = {
            urls: urls,
            depth: document.getElementById('depth').value,
            same_domain: document.getElementById('same-domain').checked
        };

        try {
            const response = await fetch('/admin/scrape', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const {value, done} = await reader.read();
                if (done) break;
                
                const events = decoder.decode(value).split('\n').filter(Boolean);
                events.forEach(event => {
                    const data = JSON.parse(event);
                    updateProgress(data);
                });
            }
        } catch (error) {
            console.error('Error:', error);
            progressText.textContent = 'Error occurred while scraping';
        }
    });
    function updateLogs() {
        fetch('/admin/scrape-logs')
            .then(response => response.json())
            .then(data => {
                const logsDiv = document.getElementById('scrape-logs');
                logsDiv.innerHTML = data.logs.map(log => 
                    `<div class="text-sm text-gray-600 dark:text-gray-400">${log}</div>`
                ).join('');
                logsDiv.scrollTop = logsDiv.scrollHeight;
            });
    }
    
    // Update logs every few seconds while scraping
    let logsInterval;
    function startLogging() {
        logsInterval = setInterval(updateLogs, 2000);
    }
    
    function stopLogging() {
        clearInterval(logsInterval);
    }
    function updateProgress(data) {
        if (data.type === 'progress') {
            progressBar.style.width = `${data.percentage}%`;
            progressText.textContent = data.message;
        } else if (data.type === 'url') {
            const li = document.createElement('li');
            li.className = 'py-3';
            li.innerHTML = `
                <div class="flex items-center space-x-3">
                    <svg class="h-5 w-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
                    </svg>
                    <span class="text-sm text-gray-900">${data.url}</span>
                </div>
            `;
            scrapedUrls.appendChild(li);
        }
    }
});