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

// function setupSearchHandler() {
//     const searchForm = document.querySelector('#search-input').closest('div');
//     const searchResults = document.getElementById('search-results');
    
//     searchForm.addEventListener('submit', async function(e) {
//         e.preventDefault();
//         const query = document.getElementById('search-input').value;
        
//         try {
//             const response = await fetch('/admin/search', {
//                 method: 'POST',
//                 headers: {
//                     'Content-Type': 'application/json',
//                 },
//                 body: JSON.stringify({ query })
//             });
            
//             const results = await response.json();
//             searchResults.innerHTML = results.map(result => `
//                 <div class="mb-4 p-4 bg-gray-50 rounded-lg">
//                     <div class="text-sm text-gray-900">${result.text}</div>
//                     <div class="mt-2 text-xs text-gray-500">
//                         Source: ${result.metadata.source}
//                     </div>
//                 </div>
//             `).join('');
//         } catch (error) {
//             console.error('Error searching:', error);
//         }
//     });
// }


function setupSearchHandler() {
    const searchForm = document.getElementById('search-form');
    const searchResults = document.getElementById('search-results');
    const searchLoader = document.getElementById('search-loader');
    const collectionSelect = document.getElementById('collection-select');
    
    // Load collections into select
    loadCollectionsForSelect();
    
    searchForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const query = document.getElementById('search-input').value;
        const selectedCollection = collectionSelect.value;
        
        if (!query.trim()) return;
        
        // Show loader
        searchLoader.classList.remove('hidden');
        searchResults.innerHTML = '';
        
        try {
            const response = await fetch('/admin/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    query,
                    collection: selectedCollection || null
                })
            });
            
            const data = await response.json();
            
            // Hide loader
            searchLoader.classList.add('hidden');
            
            if (data.results && data.results.length > 0) {
                searchResults.innerHTML = data.results.map((result, index) => `
                    <div class="result-item mb-4 p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors duration-150 ease-in-out"
                         style="animation-delay: ${index * 100}ms">
                        <div class="flex justify-between items-start">
                            <div class="flex-1">
                                <div class="text-sm text-gray-900">${result.text}</div>
                                <div class="mt-2 flex items-center space-x-4">
                                    <span class="text-xs text-gray-500">
                                        Collection: ${result.collection}
                                    </span>
                                    ${result.metadata.score ? `
                                        <span class="text-xs text-gray-500">
                                            Relevance: ${(result.metadata.score * 100).toFixed(1)}%
                                        </span>
                                    ` : ''}
                                </div>
                            </div>
                            ${result.metadata.url ? `
                                <a href="${result.metadata.url}" 
                                   target="_blank"
                                   class="ml-4 text-blue-600 hover:text-blue-800 text-sm">
                                    View Source →
                                </a>
                            ` : ''}
                        </div>
                    </div>
                `).join('');
            } else {
                searchResults.innerHTML = `
                    <div class="text-center py-8 text-gray-500">
                        No results found for "${query}"
                    </div>
                `;
            }
        } catch (error) {
            console.error('Error searching:', error);
            searchLoader.classList.add('hidden');
            searchResults.innerHTML = `
                <div class="text-center py-8 text-red-500">
                    An error occurred while searching. Please try again.
                </div>
            `;
        }
    });
}

async function loadCollectionsForSelect() {
    try {
        const response = await fetch('/admin/collections');
        const collections = await response.json();
        const select = document.getElementById('collection-select');
        
        collections.forEach(collection => {
            const option = document.createElement('option');
            option.value = collection.name;
            option.textContent = `${collection.name} (${collection.document_count} docs)`;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading collections for select:', error);
    }
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