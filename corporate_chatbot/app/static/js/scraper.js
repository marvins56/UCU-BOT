document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('scraper-form');
    const resultsPanel = document.getElementById('results-panel');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const scrapedUrls = document.getElementById('scraped-urls');
    const logsDiv = document.getElementById('scrape-logs');

    let lastLogLength = 0;
    let pollInterval = null;
    let startTime = null;
    let totalPages = 0;
    let totalLinks = 0;
    let totalChars = 0;
    let isScrapingActive = false;

    function startLogPolling() {
        return setInterval(async () => {
            if (!isScrapingActive) return;
            
            try {
                const response = await fetch('/admin/read-logs');
                const data = await response.json();
                
                if (data.logs.length > lastLogLength) {
                    const newLogs = data.logs.slice(lastLogLength);
                    newLogs.forEach(log => {
                        if (log.trim()) {  // Only add non-empty logs
                            addLog(log.trim());
                            
                            // Update stats based on log content
                            if (log.includes('Found')) {
                                const linksMatch = log.match(/Found (\d+) raw links/);
                                if (linksMatch) {
                                    totalLinks += parseInt(linksMatch[1]);
                                    updateStats();
                                }
                            }
                            if (log.includes('Extracted')) {
                                const charsMatch = log.match(/Extracted (\d+) characters/);
                                if (charsMatch) {
                                    totalChars += parseInt(charsMatch[1]);
                                    updateStats();
                                }
                            }
                            if (log.includes('Scraping URL')) {
                                totalPages++;
                                updateStats();
                            }
                        }
                    });
                    lastLogLength = data.logs.length;
                    
                    // Auto-scroll logs
                    logsDiv.scrollTop = logsDiv.scrollHeight;
                }
            } catch (error) {
                console.error('Error polling logs:', error);
            }
        }, 1000);
    }

    function updateStats() {
        const pagesElement = document.getElementById('pages-scraped');
        const linksElement = document.getElementById('links-found');
        const contentElement = document.getElementById('content-size');
        const timeElement = document.getElementById('time-elapsed');

        if (pagesElement) pagesElement.textContent = totalPages;
        if (linksElement) linksElement.textContent = totalLinks;
        if (contentElement) contentElement.textContent = `${totalChars} chars`;
        
        if (timeElement && startTime) {
            const elapsed = Math.round((Date.now() - startTime) / 1000);
            timeElement.textContent = `${elapsed}s`;
        }
    }

    function addLog(message, type = 'info', timestamp = null) {
        const log = document.createElement('div');
        log.className = `mb-1 ${type === 'error' ? 'text-red-500' : 'text-gray-600 dark:text-gray-400'}`;
        log.innerHTML = `
            <span class="opacity-75">[${timestamp || new Date().toLocaleTimeString()}]</span>
            ${message}
        `;
        logsDiv.appendChild(log);
        logsDiv.scrollTop = logsDiv.scrollHeight;
    }

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const urls = document.getElementById('urls').value.split('\n').filter(url => url.trim());
        if (urls.length === 0) {
            alert('Please enter at least one URL');
            return;
        }

        // Reset everything
        isScrapingActive = true;
        lastLogLength = 0;
        startTime = Date.now();
        totalPages = 0;
        totalLinks = 0;
        totalChars = 0;
        
        // Reset UI
        resultsPanel.style.display = 'block';
        progressBar.style.width = '0%';
        progressText.textContent = 'Starting scrape...';
        scrapedUrls.innerHTML = '';
        logsDiv.innerHTML = '';
        
        // Disable submit button
        const submitButton = this.querySelector('button[type="submit"]');
        submitButton.disabled = true;

        // Start log polling
        if (pollInterval) clearInterval(pollInterval);
        pollInterval = startLogPolling();

        try {
            const response = await fetch('/admin/scrape', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    urls: urls,
                    depth: document.getElementById('depth').value,
                    same_domain: document.getElementById('same-domain').checked
                })
            });

            const data = await response.json();
            
            // Display results
            data.results.forEach(result => {
                const li = document.createElement('li');
                li.className = 'py-3 px-4 hover:bg-gray-50 dark:hover:bg-gray-800';
                li.innerHTML = `
                    <div class="flex items-center justify-between">
                        <div class="flex items-center space-x-3">
                            <svg class="h-5 w-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
                            </svg>
                            <div>
                                <div class="text-sm font-medium text-gray-900 dark:text-white">
                                    ${result.title || result.url}
                                </div>
                                <div class="flex space-x-2">
                                    <span class="text-sm text-gray-500">${result.content.length} characters</span>
                                    <span class="text-sm text-gray-500">${result.metadata.links_found} links</span>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                scrapedUrls.appendChild(li);
            });

            progressBar.style.width = '100%';
            progressText.textContent = 'Scraping completed';

        } catch (error) {
            console.error('Error:', error);
            addLog('Error occurred while scraping', 'error');
            progressText.textContent = 'Error occurred while scraping';
        } finally {
            // Clean up
            isScrapingActive = false;
            submitButton.disabled = false;
            if (pollInterval) {
                setTimeout(() => {
                    clearInterval(pollInterval);
                    pollInterval = null;
                }, 2000); // Wait 2 seconds before stopping polling to catch final logs
            }
        }
    });

    // Update timer every second while scraping is active
    setInterval(() => {
        if (startTime && isScrapingActive) {
            updateStats();
        }
    }, 1000);
});