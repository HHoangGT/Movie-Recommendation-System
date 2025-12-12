/**
 * Production Web Application - JavaScript
 * Unified Movie Recommendation System with Role-Based Access
 */

// Get user role from session storage
const userRole = sessionStorage.getItem('userRole') || 'user';

// API Configuration
const API_BASE = '';
const USER_ID = 'web_user_' + Math.random().toString(36).substr(2, 9);

// App State
const state = {
    activeModel: 'nextitnet',
    history: [],
    recommendations: [],
    movies: [],
    userRole: userRole,
    isLoading: false
};

// ============================================
// API Functions
// ============================================

async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(API_BASE + endpoint, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

const api = {
    getHealth: () => apiCall('/api/health'),
    getActiveModel: () => apiCall('/api/model/active'),
    switchModel: (model) => apiCall('/api/model/switch', {
        method: 'POST',
        body: JSON.stringify({ model })
    }),
    getRecommendations: (userId, topK = 10) =>
        apiCall(`/api/recommendations/${userId}?top_k=${topK}`),
    getLLMRecommendations: (movieName, movieGenre, movieOverview, topK = 10) =>
        apiCall('/api/recommendations/llm', {
            method: 'POST',
            body: JSON.stringify({
                movie_name: movieName,
                movie_genre: movieGenre,
                movie_overview: movieOverview,
                top_k: topK
            })
        }),
    getHistory: (userId) => apiCall(`/api/history/${userId}`),
    addToHistory: (userId, movieId) => apiCall(`/api/history/${userId}`, {
        method: 'POST',
        body: JSON.stringify({ movie_id: movieId })
    }),
    clearHistory: (userId) => apiCall(`/api/history/${userId}`, {
        method: 'DELETE'
    }),
    removeFromHistory: (userId, movieId) => apiCall(`/api/history/${userId}/remove`, {
        method: 'POST',
        body: JSON.stringify({ movie_id: movieId })
    }),
    searchMovies: (query, limit = 10) =>
        apiCall(`/api/movies/search?q=${encodeURIComponent(query)}&limit=${limit}`),
    getMovies: (limit = 20) => apiCall(`/api/movies?limit=${limit}`),
    getMovie: (movieId) => apiCall(`/api/movies/${movieId}`),

    // Admin APIs
    getAdminStats: () => apiCall('/api/admin/stats'),
    getAllMoviesAdmin: (limit = 100, offset = 0) =>
        apiCall(`/api/admin/movies/all?limit=${limit}&offset=${offset}`),
    getAllUsersAdmin: () => apiCall('/api/admin/users')
};

// ============================================
// Role-Based UI Setup
// ============================================

function setupRoleBasedUI() {
    const roleText = document.getElementById('userRoleText');
    const adminTab = document.getElementById('adminTab');
    const dataScientistTab = document.getElementById('dataScientistTab');

    // Update role text
    const roleNames = {
        'user': 'User View (Người xem)',
        'admin': 'Admin View (Quản trị)',
        'datascientist': 'Data Scientist View (Nhà khoa học dữ liệu)'
    };
    if (roleText) roleText.textContent = roleNames[userRole] || 'User';

    // Show/hide tabs based on role
    if (userRole === 'admin') {
        if (adminTab) adminTab.classList.remove('hidden');
    } else if (userRole === 'datascientist') {
        if (adminTab) adminTab.classList.remove('hidden');
        if (dataScientistTab) dataScientistTab.classList.remove('hidden');
    }
}

// ============================================
// UI Functions
// ============================================

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type} bg-gray-800 border border-gray-600 rounded-lg p-4 text-white shadow-lg`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function updateStatus(isReady, device = 'Unknown') {
    const indicator = document.getElementById('statusIndicator');
    const text = document.getElementById('statusText');

    if (isReady) {
        indicator.className = 'w-3 h-3 rounded-full bg-green-500';
        text.textContent = `Ready (${device})`;
    } else {
        indicator.className = 'w-3 h-3 rounded-full bg-red-500';
        text.textContent = 'Disconnected';
    }
}

// function updateActiveModelBadge(model) {
//     const badge = document.getElementById('activeModelBadge');
//     if (model === 'nextitnet') {
//         badge.textContent = 'NextItNet';
//         badge.className = 'px-3 py-1 bg-emerald-600 text-white text-sm font-medium rounded-full';
//     } else {
//         badge.textContent = 'LLM (Gemini)';
//         badge.className = 'px-3 py-1 bg-purple-600 text-white text-sm font-medium rounded-full';
//     }
// }
function updateActiveModelBadge(model) {
    const badge = document.getElementById('activeModelBadge');
    if (!badge) return;

    if (model === 'nextitnet') {
        badge.textContent = 'NextItNet';
        badge.className = 'px-3 py-1 bg-emerald-600 text-white text-sm font-medium rounded-full';
    } else if (model === 'llm') {
        badge.textContent = 'LLM (Gemini)';
        badge.className = 'px-3 py-1 bg-purple-600 text-white text-sm font-medium rounded-full';
    } else if (model === 'bivae') {
        // Thêm trường hợp cho BiVAE
        badge.textContent = 'BiVAE';
        badge.className = 'px-3 py-1 bg-indigo-600 text-white text-sm font-medium rounded-full';
    }
}

function renderMovieCard(movie) {
    if (!movie) return '';
    const movieId = movie.movie_id || movie.id || 0;
    const title = movie.title || 'Unknown Movie';
    const genres = movie.genres ? (Array.isArray(movie.genres) ? movie.genres.join(', ') : movie.genres) : '';
    const inHistory = state.history.some(m => (m.movie_id || m.id) === movieId);
    const year = movie.release_date ? movie.release_date.substring(0, 4) : '';
    const inVocab = movie.in_vocabulary !== false; // Default to true for backward compatibility

    return `
        <div class="movie-card ${inHistory ? 'in-history' : ''} ${!inVocab ? 'not-in-vocab' : ''}"
             data-movie-id="${movieId}"
             data-in-vocab="${inVocab}"
             title="${!inVocab ? 'Can add to history, but model cannot recommend it (< 5 ratings)' : 'Click to add to history'}">
            <div class="title">
                ${title}
                ${!inVocab ? '<span style="font-size:0.7rem;color:#f59e0b;margin-left:4px;" title="Not in vocabulary - too few ratings">ℹ️</span>' : ''}
            </div>
            ${year ? `<div class="meta">${year}</div>` : ''}
            ${genres ? `<div class="genres">${genres}</div>` : ''}
            ${!inVocab ? '<div style="font-size:0.65rem;color:#f59e0b;margin-top:4px;">< 5 ratings (can watch, won\'t recommend)</div>' : ''}
        </div>
    `;
}

function renderRecommendationCard(rec, index) {
    const source = rec.source || state.activeModel;
    const scoreText = rec.score ? `Score: ${(rec.score * 100).toFixed(1)}%` : '';
    const overview = rec.overview || '';

    return `
        <div class="rec-card">
            <div class="flex items-start">
                <div class="rank">${index + 1}</div>
                <div class="flex-1">
                    <div class="flex items-center justify-between">
                        <div class="title">${rec.title || 'Unknown'}</div>
                        <span class="source-badge ${source}">${source === 'llm' ? 'LLM' : 'NextItNet'}</span>
                    </div>
                    ${scoreText ? `<div class="score">${scoreText}</div>` : ''}
                    ${overview ? `<div class="overview">${overview.substring(0, 150)}${overview.length > 150 ? '...' : ''}</div>` : ''}
                </div>
            </div>
        </div>
    `;
}

function renderHistoryItem(movie) {
    if (!movie) return '';
    const movieId = movie.movie_id || movie.id || 0;
    const title = movie.title || 'Unknown Movie';
    return `
        <div class="history-item" data-movie-id="${movieId}">
            <span class="title">${title}</span>
            <span class="remove-btn" onclick="removeFromHistory(${movieId})">×</span>
        </div>
    `;
}

function renderSearchResults(results) {
    const container = document.getElementById('searchResults');
    if (!results || results.length === 0) {
        container.innerHTML = '<p class="text-gray-500 text-sm">No results found</p>';
        return;
    }
    container.innerHTML = results.map(m => renderMovieCard(m)).join('');
    attachMovieCardListeners(container);
}

function renderPopularMovies(movies, append = false) {
    const container = document.getElementById('popularMovies');
    const html = movies.map(m => renderMovieCard(m)).join('');

    if (append) {
        container.innerHTML += html;
    } else {
        container.innerHTML = html;
    }
    attachMovieCardListeners(container);
}

function renderRecommendations(recommendations) {
    const container = document.getElementById('recommendations');
    const emptyState = document.getElementById('emptyRecommendations');

    if (!container) {
        console.error('Recommendations container not found');
        return;
    }

    if (!recommendations || recommendations.length === 0) {
        if (emptyState) {
            emptyState.style.display = 'block';
            container.innerHTML = '';
            container.appendChild(emptyState);
        }
        return;
    }

    if (emptyState) {
        emptyState.style.display = 'none';
    }
    container.innerHTML = recommendations.map((rec, i) => renderRecommendationCard(rec, i)).join('');
}

function renderHistory(history) {
    const container = document.getElementById('historyList');
    const emptyState = document.getElementById('emptyHistory');
    const count = document.getElementById('historyCount');

    if (!container) return;

    if (count) {
        count.textContent = Array.isArray(history) ? history.length : 0;
    }

    if (!history || history.length === 0) {
        // Show empty state, remove all movie items
        if (emptyState) {
            emptyState.style.display = 'block';
        }
        // Remove only movie items, not the emptyHistory element
        const movieItems = container.querySelectorAll('.history-item');
        movieItems.forEach(item => item.remove());
        return;
    }

    // Hide empty state, show movies
    if (emptyState) {
        emptyState.style.display = 'none';
    }
    // Create a temporary container for the movie items
    const moviesHTML = history.map(m => renderHistoryItem(m)).join('');
    // Remove existing movie items
    const existingItems = container.querySelectorAll('.history-item');
    existingItems.forEach(item => item.remove());
    // Insert new movie items (keeping emptyHistory in place)
    container.insertAdjacentHTML('beforeend', moviesHTML);
}

function attachMovieCardListeners(container) {
    container.querySelectorAll('.movie-card').forEach(card => {
        card.addEventListener('click', () => {
            const movieId = parseInt(card.dataset.movieId);
            const inVocab = card.dataset.inVocab === 'true';

            // Info message if movie is not in vocabulary (but still allow adding)
            if (!inVocab) {
                showToast('ℹ️ Added to history, but won\'t affect recommendations (too few ratings)', 'info');
            }

            addToHistory(movieId);
        });
    });
}

// ============================================
// Model Switching (Data Scientist Only)
// ============================================

// function updateModelCards() {
//     const nextitnetCard = document.getElementById('modelCard-nextitnet');
//     const llmCard = document.getElementById('modelCard-llm');

//     if (!nextitnetCard || !llmCard) return;

//     // Update cards
//     if (state.activeModel === 'nextitnet') {
//         nextitnetCard.className = 'model-card border-2 border-emerald-500 bg-emerald-50 rounded-xl p-4 cursor-pointer transition hover:shadow-md';
//         llmCard.className = 'model-card border-2 border-gray-200 bg-white rounded-xl p-4 cursor-pointer transition hover:shadow-md';

//         document.getElementById('status-nextitnet').textContent = 'Active';
//         document.getElementById('status-nextitnet').className = 'px-2 py-1 bg-emerald-100 text-emerald-700 text-xs font-medium rounded-full';
//         document.getElementById('status-llm').textContent = 'Standby';
//         document.getElementById('status-llm').className = 'px-2 py-1 bg-gray-100 text-gray-600 text-xs font-medium rounded-full';
//     } else {
//         llmCard.className = 'model-card border-2 border-purple-500 bg-purple-50 rounded-xl p-4 cursor-pointer transition hover:shadow-md';
//         nextitnetCard.className = 'model-card border-2 border-gray-200 bg-white rounded-xl p-4 cursor-pointer transition hover:shadow-md';

//         document.getElementById('status-llm').textContent = 'Active';
//         document.getElementById('status-llm').className = 'px-2 py-1 bg-purple-100 text-purple-700 text-xs font-medium rounded-full';
//         document.getElementById('status-nextitnet').textContent = 'Standby';
//         document.getElementById('status-nextitnet').className = 'px-2 py-1 bg-gray-100 text-gray-600 text-xs font-medium rounded-full';
//     }

//     // Update Model Lab
//     updateModelLab();
// }
function updateModelCards() {
    const models = ['nextitnet', 'llm', 'bivae']; // Danh sách model

    // Config màu sắc cho từng model
    const styles = {
        'nextitnet': { color: 'emerald', bg: 'emerald-50', border: 'emerald-500', text: 'emerald-700', bgBadge: 'emerald-100' },
        'llm': { color: 'purple', bg: 'purple-50', border: 'purple-500', text: 'purple-700', bgBadge: 'purple-100' },
        'bivae': { color: 'indigo', bg: 'indigo-50', border: 'indigo-500', text: 'indigo-700', bgBadge: 'indigo-100' } // Màu cho BiVAE
    };

    models.forEach(model => {
        const card = document.getElementById(`modelCard-${model}`);
        const status = document.getElementById(`status-${model}`);

        if (!card) return; // Bỏ qua nếu card không tồn tại trên HTML

        if (state.activeModel === model) {
            // Active Style
            const s = styles[model];
            card.className = `model-card border-2 border-${s.border} bg-${s.bg} rounded-xl p-4 cursor-pointer transition hover:shadow-md`;
            if (status) {
                status.textContent = 'Active';
                status.className = `px-2 py-1 bg-${s.bgBadge} text-${s.text} text-xs font-medium rounded-full`;
            }
        } else {
            // Standby Style
            card.className = 'model-card border-2 border-gray-200 bg-white rounded-xl p-4 cursor-pointer transition hover:shadow-md hover:border-gray-300';
            if (status) {
                status.textContent = 'Standby';
                status.className = 'px-2 py-1 bg-gray-100 text-gray-600 text-xs font-medium rounded-full';
            }
        }
    });

    // Update Model Lab sub-function
    updateModelLab();
}

// function updateModelLab() {
//     const activeModelInfo = document.getElementById('activeModelInfo');
//     const labActiveModelName = document.getElementById('labActiveModelName');
//     const labActiveModelBadge = document.getElementById('labActiveModelBadge');
//     const labActiveModelDesc = document.getElementById('labActiveModelDesc');
//     const nextitnetDetails = document.getElementById('nextitnetDetails');
//     const llmDetails = document.getElementById('llmDetails');
//     const nextitnetBadge = document.getElementById('nextitnetBadge');
//     const llmBadge = document.getElementById('llmBadge');

//     if (state.activeModel === 'nextitnet') {
//         if (activeModelInfo) {
//             activeModelInfo.className = 'bg-white p-6 rounded-xl shadow-sm border-l-4 border-emerald-500';
//         }
//         if (labActiveModelName) labActiveModelName.textContent = 'NextItNet';
//         if (labActiveModelDesc) labActiveModelDesc.textContent = 'Sequential recommendation model using dilated causal convolutions';
//         if (nextitnetDetails) nextitnetDetails.open = true;
//         if (llmDetails) llmDetails.open = false;
//         if (nextitnetBadge) {
//             nextitnetBadge.textContent = 'Active';
//             nextitnetBadge.className = 'px-3 py-0.5 rounded-full text-sm font-medium bg-emerald-100 text-emerald-700';
//         }
//         if (llmBadge) {
//             llmBadge.textContent = 'Standby';
//             llmBadge.className = 'px-3 py-0.5 rounded-full text-sm font-medium bg-gray-100 text-gray-700';
//         }
//     } else {
//         if (activeModelInfo) {
//             activeModelInfo.className = 'bg-white p-6 rounded-xl shadow-sm border-l-4 border-purple-500';
//         }
//         if (labActiveModelName) labActiveModelName.textContent = 'LLM (Gemini)';
//         if (labActiveModelDesc) labActiveModelDesc.textContent = 'AI-powered content-based recommendations using OpenRouter';
//         if (llmDetails) llmDetails.open = true;
//         if (nextitnetDetails) nextitnetDetails.open = false;
//         if (llmBadge) {
//             llmBadge.textContent = 'Active';
//             llmBadge.className = 'px-3 py-0.5 rounded-full text-sm font-medium bg-purple-100 text-purple-700';
//         }
//         if (nextitnetBadge) {
//             nextitnetBadge.textContent = 'Standby';
//             nextitnetBadge.className = 'px-3 py-0.5 rounded-full text-sm font-medium bg-gray-100 text-gray-700';
//         }
//     }

//     // Update DS dashboard
//     const dsActiveModel = document.getElementById('dsActiveModel');
//     const adminActiveModel = document.getElementById('adminActiveModel');
//     const modelText = state.activeModel === 'nextitnet' ? 'NextItNet' : 'LLM (Gemini)';
//     if (dsActiveModel) dsActiveModel.textContent = modelText;
//     if (adminActiveModel) adminActiveModel.textContent = modelText;
// }

function updateModelLab() {
    const activeModelInfo = document.getElementById('activeModelInfo');
    const labActiveModelName = document.getElementById('labActiveModelName');
    const labActiveModelDesc = document.getElementById('labActiveModelDesc');

    // Details elements
    const detailsMap = {
        'nextitnet': document.getElementById('nextitnetDetails'),
        'llm': document.getElementById('llmDetails'),
        'bivae': document.getElementById('bivaeDetails')
    };

    // Reset UI state
    if (activeModelInfo) activeModelInfo.className = 'bg-white p-6 rounded-xl shadow-sm border-l-4 border-gray-500';
    Object.values(detailsMap).forEach(el => { if(el) el.open = false; });

    // Update based on active model
    if (state.activeModel === 'nextitnet') {
        if (activeModelInfo) activeModelInfo.className = 'bg-white p-6 rounded-xl shadow-sm border-l-4 border-emerald-500';
        if (labActiveModelName) labActiveModelName.textContent = 'NextItNet';
        if (labActiveModelDesc) labActiveModelDesc.textContent = 'Sequential recommendation model using dilated causal convolutions';
        if (detailsMap.nextitnet) detailsMap.nextitnet.open = true;

    } else if (state.activeModel === 'llm') {
        if (activeModelInfo) activeModelInfo.className = 'bg-white p-6 rounded-xl shadow-sm border-l-4 border-purple-500';
        if (labActiveModelName) labActiveModelName.textContent = 'LLM (Gemini)';
        if (labActiveModelDesc) labActiveModelDesc.textContent = 'AI-powered content-based recommendations using OpenRouter';
        if (detailsMap.llm) detailsMap.llm.open = true;

    } else if (state.activeModel === 'bivae') {
        if (activeModelInfo) activeModelInfo.className = 'bg-white p-6 rounded-xl shadow-sm border-l-4 border-indigo-500';
        if (labActiveModelName) labActiveModelName.textContent = 'BiVAE';
        if (labActiveModelDesc) labActiveModelDesc.textContent = 'Generative collaborative filtering using Variational Autoencoder';
        if (detailsMap.bivae) detailsMap.bivae.open = true;
    }

    // Update Badges on Dashboard (Admin/DS View)
    const dsActiveModel = document.getElementById('dsActiveModel');
    const adminActiveModel = document.getElementById('adminActiveModel');
    const modelTexts = { 'nextitnet': 'NextItNet', 'llm': 'LLM (Gemini)', 'bivae': 'BiVAE' };

    if (dsActiveModel) dsActiveModel.textContent = modelTexts[state.activeModel];
    if (adminActiveModel) adminActiveModel.textContent = modelTexts[state.activeModel];
}

async function switchModel(model) {
    // Check if user has permission
    if (userRole !== 'datascientist') {
        showToast('Only Data Scientists can switch models', 'error');
        return;
    }

    try {
        const response = await api.switchModel(model);
        if (response.success) {
            state.activeModel = model;
            updateActiveModelBadge(model);
            updateModelCards();
            const modelNames = {
                'nextitnet': 'NextItNet',
                'llm': 'LLM (Gemini)',
                'bivae': 'BiVAE'
            };
            // showToast(`Switched to ${model === 'nextitnet' ? 'NextItNet' : 'LLM (Gemini)'}`, 'success');
            showToast(`Switched to ${modelNames[model] || model}`, 'success');
        } else {
            showToast(response.message || 'Failed to switch model', 'error');
        }
    } catch (error) {
        showToast('Failed to switch model', 'error');
    }
}

// ============================================
// Data Functions
// ============================================

async function loadInitialData() {
    try {
        const health = await api.getHealth();

        updateStatus(health.is_ready, health.device);
        state.activeModel = health.active_model || 'nextitnet';
        updateActiveModelBadge(state.activeModel);

        // Update DS dashboard
        if (document.getElementById('dsDeviceStatus')) {
            document.getElementById('dsDeviceStatus').textContent = health.device || 'CPU';
            document.getElementById('dsApiStatus').textContent = 'Connected';
            document.getElementById('dsMovieCount').textContent = (health.num_movies || 3497).toLocaleString();
            document.getElementById('dsSessionCount').textContent = health.num_sessions || 0;
        }

        // Update model cards
        updateModelCards();

        // Load ALL movies from dataset (9,066 movies total)
        const moviesResponse = await api.getMovies(10000); // Load all movies
        if (moviesResponse.success) {
            state.movies = moviesResponse.movies;
            renderPopularMovies(state.movies);
            const vocabCount = state.movies.filter(m => m.in_vocabulary).length;
            console.log(`✅ Loaded ${state.movies.length} total movies (${vocabCount} in vocabulary, can be recommended)`);
        }

        // Load history
        const historyResponse = await api.getHistory(USER_ID);
        if (historyResponse.success) {
            state.history = historyResponse.history || [];
            renderHistory(state.history);
        }

        // Load admin data if admin
        if (userRole === 'admin' || userRole === 'datascientist') {
            loadAdminData();
        }

    } catch (error) {
        console.error('Failed to load initial data:', error);
        updateStatus(false);
        showToast('Failed to connect to API', 'error');
    }
}

async function searchMovies() {
    const query = document.getElementById('searchInput').value.trim();
    if (!query) return;

    try {
        const response = await api.searchMovies(query);
        if (response.success) {
            renderSearchResults(response.results);
        }
    } catch (error) {
        showToast('Search failed', 'error');
    }
}

async function addToHistory(movieId) {
    try {
        console.log('Adding movie to history:', movieId);
        const response = await api.addToHistory(USER_ID, movieId);

        if (response && response.success) {
            const historyResponse = await api.getHistory(USER_ID);
            if (historyResponse && historyResponse.success) {
                state.history = historyResponse.history || [];
                renderHistory(state.history);
            }
            showToast('Added to history', 'success');
        } else {
            showToast((response && response.message) || 'Failed to add', 'error');
        }
    } catch (error) {
        console.error('Add to history error:', error);
        showToast('Failed to add to history', 'error');
    }
}

async function removeFromHistory(movieId) {
    try {
        const response = await api.removeFromHistory(USER_ID, movieId);
        if (response.success) {
            const historyResponse = await api.getHistory(USER_ID);
            if (historyResponse.success) {
                state.history = historyResponse.history || [];
                renderHistory(state.history);
            }
            showToast('Removed from history', 'success');
        } else {
            showToast(response.message || 'Failed to remove', 'error');
        }
    } catch (error) {
        showToast('Failed to remove from history', 'error');
    }
}

async function clearHistory() {
    if (!confirm('Clear all history?')) return;

    try {
        const response = await api.clearHistory(USER_ID);
        if (response && response.success) {
            state.history = [];
            renderHistory([]);
            renderRecommendations([]);
            showToast('History cleared', 'success');
        } else {
            showToast((response && response.message) || 'Failed to clear', 'error');
        }
    } catch (error) {
        showToast('Failed to clear history', 'error');
    }
}

async function getRecommendations() {
    try {
        const historyResponse = await api.getHistory(USER_ID);
        if (historyResponse.success) {
            state.history = historyResponse.history || [];
        }

        if (state.history.length === 0) {
            showToast('Add some movies to your history first', 'info');
            return;
        }

        const response = await api.getRecommendations(USER_ID);
        if (response.success) {
            state.recommendations = response.recommendations;
            renderRecommendations(state.recommendations);
            showToast(`Got ${state.recommendations.length} recommendations (${response.model_used})`, 'success');
        } else {
            showToast(response.message || 'Failed to get recommendations', 'error');
        }
    } catch (error) {
        console.error('Get recommendations error:', error);
        showToast('Failed to get recommendations', 'error');
    }
}

async function loadMoreMovies() {
    // All movies should already be loaded on initial load
    const totalMovies = 9066; // Total movies in dataset (from EDA)
    const currentCount = state.movies.length;

    if (currentCount >= totalMovies || currentCount >= 9000) {
        showToast(`All ${currentCount} movies are already loaded!`, 'info');
        return;
    }

    try {
        const response = await api.getMovies(10000); // Load all
        if (response.success && response.movies.length > currentCount) {
            const newMovies = response.movies.slice(currentCount);
            state.movies = response.movies;
            renderPopularMovies(newMovies, true);
            const vocabCount = state.movies.filter(m => m.in_vocabulary).length;
            showToast(`Loaded ${newMovies.length} more movies (Total: ${state.movies.length}, ${vocabCount} in vocabulary)`, 'success');
        } else {
            showToast(`All ${currentCount} movies loaded!`, 'info');
        }
    } catch (error) {
        showToast('Failed to load more movies', 'error');
    }
}

// ============================================
// Admin Functions
// ============================================

async function loadAdminData() {
    try {
        const stats = await api.getAdminStats();
        if (stats.success) {
            const s = stats.stats;
            if (document.getElementById('adminTotalMovies')) {
                document.getElementById('adminTotalMovies').textContent = s.total_movies.toLocaleString();
            }
            if (document.getElementById('adminTotalUsers')) {
                document.getElementById('adminTotalUsers').textContent = s.total_sessions.toLocaleString();
            }
        }

        // Load movies for admin table
        const movies = await api.getAllMoviesAdmin(50, 0);
        if (movies.success) {
            renderAdminMoviesTable(movies.movies);
        }

        // Load users
        const users = await api.getAllUsersAdmin();
        if (users.success) {
            renderAdminUsersTable(users.users);
        }
    } catch (error) {
        console.error('Failed to load admin data:', error);
    }
}

function renderAdminMoviesTable(movies) {
    const table = document.getElementById('adminMoviesTable');
    if (!table) return;

    table.innerHTML = movies.slice(0, 10).map(m => `
        <tr>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${m.movie_id || m.id}</td>
            <td class="px-6 py-4 text-sm text-gray-900">${m.title || 'Unknown'}</td>
            <td class="px-6 py-4 text-sm text-gray-500">${Array.isArray(m.genres) ? m.genres.slice(0,2).join(', ') : ''}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${m.release_date ? m.release_date.substring(0,4) : '-'}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">${m.vote_average || '-'}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">-</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm">
                <button class="text-blue-600 hover:text-blue-900 mr-3">✏️</button>
                <button class="text-red-600 hover:text-red-900">🗑️</button>
            </td>
        </tr>
    `).join('');
}

function renderAdminUsersTable(users) {
    const table = document.getElementById('adminUsersTable');
    if (!table) return;

    table.innerHTML = users.map(u => `
        <tr>
            <td class="px-6 py-4 text-sm text-gray-900">${u.user_id}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${u.history_length}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${u.last_active}</td>
            <td class="px-6 py-4 whitespace-nowrap text-sm">
                <button class="text-blue-600 hover:text-blue-900 mr-3">View</button>
                <button class="text-red-600 hover:text-red-900">Clear</button>
            </td>
        </tr>
    `).join('');
}

// ============================================
// LLM Test
// ============================================

async function testLLM() {
    const movieName = document.getElementById('llmTestMovie')?.value.trim();
    const movieGenre = document.getElementById('llmTestGenre')?.value.trim();
    const movieOverview = document.getElementById('llmTestOverview')?.value.trim();
    const resultsDiv = document.getElementById('llmTestResults');

    if (!movieName) {
        showToast('Please enter a movie name', 'error');
        return;
    }

    resultsDiv.innerHTML = '<div class="text-gray-500">Loading...</div>';

    try {
        const response = await api.getLLMRecommendations(movieName, movieGenre, movieOverview);

        if (response.success) {
            resultsDiv.innerHTML = response.recommendations.map((rec, i) => `
                <div class="mb-3 p-2 bg-white rounded border border-gray-200">
                    <div class="font-medium text-gray-900">${i + 1}. ${rec.title}</div>
                    ${rec.genres ? `<div class="text-xs text-gray-500">${Array.isArray(rec.genres) ? rec.genres.join(', ') : rec.genres}</div>` : ''}
                    ${rec.overview ? `<div class="text-xs text-gray-600 mt-1">${rec.overview}</div>` : ''}
                </div>
            `).join('') || '<p class="text-gray-500">No recommendations returned</p>';
        } else {
            resultsDiv.innerHTML = `<p class="text-red-500">${response.message}</p>`;
        }
    } catch (error) {
        resultsDiv.innerHTML = `<p class="text-red-500">Error: ${error.message}</p>`;
    }
}

// ============================================
// Event Listeners
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    // Setup role-based UI
    setupRoleBasedUI();

    // Initialize Lucide icons
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }

    // Load initial data
    loadInitialData();

    // Main tab switching
    document.querySelectorAll('.main-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const targetId = tab.dataset.tab;

            document.querySelectorAll('.main-tab').forEach(t => {
                t.classList.remove('active', 'border-amber-400', 'text-amber-400');
                t.classList.add('border-transparent', 'text-gray-400');
            });
            tab.classList.add('active', 'border-amber-400', 'text-amber-400');

            document.querySelectorAll('.main-content').forEach(content => {
                content.classList.toggle('active', content.id === `tab-${targetId}`);
                content.classList.toggle('hidden', content.id !== `tab-${targetId}`);
            });

            setTimeout(() => {
                if (typeof lucide !== 'undefined') lucide.createIcons();
            }, 100);
        });
    });

    // DS sub-tabs
    document.querySelectorAll('.ds-nav-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const targetPageId = tab.dataset.dsTab;

            document.querySelectorAll('.ds-nav-tab').forEach(t => {
                t.classList.remove('border-emerald-500', 'text-emerald-600', 'font-semibold');
                t.classList.add('border-transparent', 'text-gray-500');
            });
            tab.classList.add('border-emerald-500', 'text-emerald-600', 'font-semibold');

            document.querySelectorAll('.ds-page-content').forEach(page => {
                page.classList.toggle('hidden', page.id !== `ds-page-${targetPageId}`);
            });
        });
    });

    // Admin sub-tabs
    document.querySelectorAll('.admin-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const targetPageId = tab.dataset.adminTab;

            document.querySelectorAll('.admin-tab').forEach(t => {
                t.classList.remove('active', 'border-blue-600', 'text-blue-600');
                t.classList.add('border-transparent', 'text-gray-600');
            });
            tab.classList.add('active', 'border-blue-600', 'text-blue-600');

            document.querySelectorAll('.admin-page-content').forEach(page => {
                page.classList.toggle('hidden', page.id !== `admin-page-${targetPageId}`);
            });
        });
    });

    // Search
    document.getElementById('searchBtn')?.addEventListener('click', searchMovies);
    document.getElementById('searchInput')?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') searchMovies();
    });

    // History actions
    document.getElementById('clearHistoryBtn')?.addEventListener('click', clearHistory);
    document.getElementById('getRecsBtn')?.addEventListener('click', getRecommendations);
    document.getElementById('refreshRecsBtn')?.addEventListener('click', getRecommendations);

    // Load more
    document.getElementById('loadMoreBtn')?.addEventListener('click', loadMoreMovies);

    // Model switching (only for data scientist)
    if (userRole === 'datascientist') {
        document.getElementById('modelCard-nextitnet')?.addEventListener('click', () => switchModel('nextitnet'));
        document.getElementById('modelCard-llm')?.addEventListener('click', () => switchModel('llm'));
        document.getElementById('modelCard-bivae')?.addEventListener('click', () => switchModel('bivae'));
    }

    // LLM test
    document.getElementById('testLlmBtn')?.addEventListener('click', testLLM);

    // Details toggle
    document.querySelectorAll('details').forEach(detail => {
        detail.addEventListener('toggle', () => {
            const icon = detail.querySelector('[data-lucide="chevron-down"]');
            if (icon) {
                icon.style.transform = detail.open ? 'rotate(180deg)' : 'rotate(0deg)';
            }
        });
    });
});
