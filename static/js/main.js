// main.js - Enhanced Frontend JavaScript for MedVQA+

const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const previewImage = document.getElementById('previewImage');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const changeImageBtn = document.getElementById('changeImageBtn');
const questionInput = document.getElementById('questionInput');
const charCounter = document.getElementById('charCounter');
const submitBtn = document.getElementById('submitBtn');
const resultSection = document.getElementById('resultSection');
const errorSection = document.getElementById('errorSection');

// Upload area click handler (only if no image)
uploadArea.addEventListener('click', (e) => {
    if (e.target !== changeImageBtn && !changeImageBtn.contains(e.target)) {
        if (previewImage.style.display === 'none') {
            imageInput.click();
        }
    }
});

// Change image button
changeImageBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    imageInput.click();
});

// Drag and drop handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--primary)';
    uploadArea.style.background = 'var(--gray-100)';
});

uploadArea.addEventListener('dragleave', () => {
    if (previewImage.style.display === 'none') {
        uploadArea.style.borderColor = 'var(--gray-300)';
        uploadArea.style.background = 'var(--gray-50)';
    }
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--gray-300)';
    uploadArea.style.background = 'var(--gray-50)';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleImageFile(files[0]);
    }
});

// Image input change handler
imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleImageFile(e.target.files[0]);
    }
});

function handleImageFile(file) {
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF, or BMP).');
        return;
    }
    
    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size too large. Maximum size is 16MB.');
        return;
    }
    
    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.style.display = 'block';
        uploadPlaceholder.style.display = 'none';
        changeImageBtn.style.display = 'flex';
        checkFormReady();
    };
    reader.readAsDataURL(file);
}

// Question input handler with character counter
questionInput.addEventListener('input', () => {
    const length = questionInput.value.length;
    charCounter.textContent = `${length} character${length !== 1 ? 's' : ''}`;
    checkFormReady();
});

function checkFormReady() {
    const hasImage = previewImage.style.display === 'block';
    const hasQuestion = questionInput.value.trim().length > 0;
    submitBtn.disabled = !(hasImage && hasQuestion);
}

// Submit handler
submitBtn.addEventListener('click', async () => {
    if (submitBtn.disabled) return;
    
    const formData = new FormData();
    formData.append('image', imageInput.files[0]);
    formData.append('question', questionInput.value.trim());
    formData.append('modality', document.getElementById('modalitySelect').value);
    
    // Show loading state
    submitBtn.disabled = true;
    submitBtn.querySelector('.btn-content').style.display = 'none';
    submitBtn.querySelector('.btn-loader').style.display = 'flex';
    hideError();
    hideResult();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Server error');
        }
        
        displayResult(data);
        
    } catch (error) {
        showError(error.message || 'An error occurred. Please try again.');
    } finally {
        // Reset button state
        submitBtn.disabled = false;
        submitBtn.querySelector('.btn-content').style.display = 'flex';
        submitBtn.querySelector('.btn-loader').style.display = 'none';
    }
});

function displayResult(data) {
    // Set question type badge
    const questionTypeBadge = document.getElementById('questionTypeBadge');
    questionTypeBadge.textContent = data.question_type === 'closed' 
        ? '✓ Closed Question (Yes/No)' 
        : '✓ Open-Ended Question';
    
    // Set model badge
    const modelBadge = document.getElementById('modelBadge');
    modelBadge.textContent = data.model_used;
    
    // Set answer
    document.getElementById('answerText').textContent = data.predicted_answer;
    
    // Set gate values with animation
    const alphaImg = data.gate_values.alpha_img;
    const alphaTxt = data.gate_values.alpha_txt;
    
    setTimeout(() => {
        document.getElementById('imageBar').style.width = `${alphaImg * 100}%`;
        document.getElementById('imageValue').textContent = `${(alphaImg * 100).toFixed(1)}%`;
        
        document.getElementById('textBar').style.width = `${alphaTxt * 100}%`;
        document.getElementById('textValue').textContent = `${(alphaTxt * 100).toFixed(1)}%`;
    }, 100);
    
    // Set dominant modality
    const gateDominant = document.getElementById('gateDominant');
    const dominant = data.gate_values.dominant;
    gateDominant.textContent = `Model relied more on ${dominant} features (${dominant === 'image' ? 'visual' : 'textual'})`;
    
    // Set RAG context if available
    const ragSection = document.getElementById('ragSection');
    const ragContext = document.getElementById('ragContext');
    
    if (data.rag_context && data.rag_context.length > 0) {
        ragContext.innerHTML = data.rag_context.map((item, idx) => `
            <div class="rag-item">
                <div class="source">[${idx + 1}] Source: ${item.source}</div>
                <div class="text">${item.text}</div>
            </div>
        `).join('');
        ragSection.style.display = 'block';
    } else {
        ragSection.style.display = 'none';
    }
    
    // Set semantic matching info if available
    const semanticSection = document.getElementById('semanticSection');
    const semanticContent = document.getElementById('semanticContent');
    
    if (data.semantic_matching) {
        const sm = data.semantic_matching;
        semanticContent.innerHTML = `
            <p><strong>Original Prediction:</strong> ${sm.original_prediction}</p>
            <p><strong>Semantic Match:</strong> ${sm.semantic_match}</p>
            <p><strong>Similarity Score:</strong> ${sm.similarity_score}</p>
            <p><strong>Candidates Checked:</strong> ${sm.candidates_checked}</p>
        `;
        semanticSection.style.display = 'block';
    } else {
        semanticSection.style.display = 'none';
    }
    
    // Show result section
    resultSection.style.display = 'block';
    
    // Scroll to result with smooth animation
    setTimeout(() => {
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    errorSection.style.display = 'block';
    resultSection.style.display = 'none';
    
    // Scroll to error
    setTimeout(() => {
        errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

function hideError() {
    errorSection.style.display = 'none';
}

function hideResult() {
    resultSection.style.display = 'none';
}

// Check health on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        console.log('Server health:', data);
        
        // Update status badge
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.header-badge span:last-child');
        
        if (data.closed_model_loaded && data.topk_model_loaded && data.rag_loaded) {
            statusDot.style.background = 'var(--secondary)';
            statusText.textContent = 'System Ready';
        } else {
            statusDot.style.background = 'var(--warning)';
            statusText.textContent = 'Models Loading...';
        }
        
        if (!data.closed_model_loaded || !data.topk_model_loaded) {
            showError('Warning: Some models are not loaded. Please check server logs.');
        }
    } catch (error) {
        console.error('Health check failed:', error);
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.header-badge span:last-child');
        statusDot.style.background = 'var(--danger)';
        statusText.textContent = 'Connection Error';
    }
});

// Add smooth scroll behavior
document.documentElement.style.scrollBehavior = 'smooth';
