document.addEventListener('DOMContentLoaded', function() {
    loadDashboardStats();
    loadCollections();
    setupSearchHandler();
    setupModalHandlers();
});

async function loadDashboardStats() {
    try {
        const response = await fetch('/admin/stats');
        const stats = await response.json();
        
        document.getElementById('total-docs').textContent = stats.total_documents;
        document.getElementById('total-collections').textContent = stats.total_collections;
        document.getElementById('storage-used').textContent = `${stats.storage_used} MB`;
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

async function loadCollections() {
    try {
        const response = await fetch('/admin/collections');
        const collections = await response.json();
        
        const tableBody = document.getElementById('collections-table');
        tableBody.innerHTML = collections.map(collection => `
            <tr>
                <td class="px-6 py-4 whitespace-nowrap">
                    <div class="text-sm font-medium text-gray-900">${collection.name}</div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                    <div class="text-sm text-gray-500">${collection.document_count}</div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                    <div class="text-sm text-gray-500">${new Date(collection.last_updated).toLocaleDateString()}</div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <button class="text-blue-600 hover:text-blue-900 mr-4">View</button>
                    <button class="text-red-600 hover:text-red-900">Delete</button>
                </td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('Error loading collections:', error);
    }
}

function setupSearchHandler() {
    const searchForm = document.querySelector('#search-input').closest('div');
    const searchResults = document.getElementById('search-results');
    
    searchForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const query = document.getElementById('search-input').value;
        
        try {
            const response = await fetch('/admin/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query })
            });
            
            const results = await response.json();
            searchResults.innerHTML = results.map(result => `
                <div class="mb-4 p-4 bg-gray-50 rounded-lg">
                    <div class="text-sm text-gray-900">${result.text}</div>
                    <div class="mt-2 text-xs text-gray-500">
                        Source: ${result.metadata.source}
                    </div>
                </div>
            `).join('');
        } catch (error) {
            console.error('Error searching:', error);
        }
    });
}

function setupModalHandlers() {
    const modal = document.getElementById('new-collection-modal');
    const openButton = document.querySelector('button:contains("New Collection")');
    const closeButton = modal.querySelector('button:contains("Cancel")');
    
    openButton.addEventListener('click', () => {
        modal.classList.remove('hidden');
    });
    
    closeButton.addEventListener('click', () => {
        modal.classList.add('hidden');
    });
    
    modal.querySelector('form').addEventListener('submit', async function(e) {
        e.preventDefault();
        const name = this.querySelector('input').value;
        
        try {
            await fetch('/admin/collections', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ name })
            });
            
            modal.classList.add('hidden');
            loadCollections();
        } catch (error) {
            console.error('Error creating collection:', error);
        }
    });
}