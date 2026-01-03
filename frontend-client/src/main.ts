const promptInput = document.querySelector<HTMLTextAreaElement>('#prompt-input');
const widthInput = document.querySelector<HTMLInputElement>('#width-input');
const heightInput = document.querySelector<HTMLInputElement>('#height-input');
const initImageInput = document.querySelector<HTMLInputElement>('#init-image'); // New file input
const clearImageButton = document.querySelector<HTMLButtonElement>('#clear-image-button'); // New clear button
const deleteHistoryButton = document.querySelector<HTMLButtonElement>('#delete-history-button'); // New delete button
const generateNewButton = document.querySelector<HTMLButtonElement>('#generete-new-button'); // New page
const generateButton = document.querySelector<HTMLButtonElement>('#generate-button');
const cancelButton = document.querySelector<HTMLButtonElement>('#cancel-button');
const downloadButton = document.querySelector<HTMLButtonElement>('#download-button');
const imageDisplay = document.querySelector<HTMLDivElement>('#image-display');
const historyList = document.querySelector<HTMLDivElement>('#history-list'); // Changed from historyGrid to historyList
const negativePromptInput = document.querySelector<HTMLTextAreaElement>('#negative-prompt');
const guidanceScaleInput = document.querySelector<HTMLInputElement>('#guidance-scale');
const guidanceScaleValue = document.querySelector<HTMLSpanElement>('#guidance-scale-value');
const inferenceStepsInput = document.querySelector<HTMLInputElement>('#inference-steps');
const inferenceStepsValue = document.querySelector<HTMLSpanElement>('#inference-steps-value');
const progressContainer = document.querySelector<HTMLDivElement>('#progress-container');
const progressBar = document.querySelector<HTMLDivElement>('#progress-bar');
const progressText = document.querySelector<HTMLDivElement>('#progress-text');
const modelSelector = document.querySelector<HTMLSelectElement>('#model-selector');
const loadingOverlay = document.querySelector<HTMLDivElement>('#loading-overlay');

// Assuming the backend is running on http://localhost:8000
const BACKEND_URL = '/api'; // Updated to use the Vite proxy
const OUTPUTS_URL = '/outputs'; // Base URL for saved images

let lastImageData: string | null = null;
let abortController: AbortController | null = null; // To manage ongoing fetch requests

// Function to fetch and render history
const fetchAndRenderHistory = async () => {
    if (!historyList) return;

    try {
        const response = await fetch(`${BACKEND_URL}/api/history`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        console.log('Received history data:', data); // Added for debugging

        historyList.innerHTML = ''; // Clear existing history

        // Handle both old format (array of strings) and new format (array of objects)
        data.history.forEach((item: any) => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';

            let filename = '';
            let prompt = '';
            let id = '';

            if (typeof item === 'string') {
                // Handle old format
                filename = item;
                prompt = item.replace('.png', '').replace(/_/g, ' ');
                id = filename; // Fallback
            } else {
                // Handle new format
                filename = item.filename;
                prompt = item.prompt;
                id = item.id;
            }

            const promptSpan = document.createElement('span');
            promptSpan.className = 'history-item-prompt';
            promptSpan.textContent = prompt;
            promptSpan.title = prompt; // Show full prompt on hover

            const deleteButton = document.createElement('button');
            deleteButton.className = 'history-item-delete-button';
            deleteButton.textContent = '×';
            deleteButton.title = 'Delete this item';

            deleteButton.addEventListener('click', async (e) => {
                e.stopPropagation(); // Prevent the history item's click event from firing
                if (confirm(`Are you sure you want to delete the image for prompt: "${prompt}"?`)) {
                    try {
                        const deleteResponse = await fetch(`${BACKEND_URL}/api/history/item/${id}`, { method: 'DELETE' }); // Corrected URL
                        if (!deleteResponse.ok) {
                            throw new Error(`HTTP error! status: ${deleteResponse.status}`);
                        }
                        historyItem.remove(); // Remove item from the DOM for immediate feedback
                    } catch (error) {
                        console.error('Error deleting history item:', error);
                        alert('Failed to delete history item.');
                    }
                }
            });

            historyItem.appendChild(promptSpan);
            historyItem.appendChild(deleteButton);

            historyItem.addEventListener('click', () => {
                // Remove .selected from all other items
                document.querySelectorAll('.history-item').forEach(i => i.classList.remove('selected'));
                // Add .selected to the clicked item
                historyItem.classList.add('selected');

                const imageUrl = `${OUTPUTS_URL}/${filename}`;
                imageDisplay!.innerHTML = `<img src="${imageUrl}" alt="${prompt}">`;
                lastImageData = imageUrl; // Set for download
                downloadButton!.style.display = 'inline-block';

                // Restore settings from history
                if (promptInput) promptInput.value = prompt;

                if (typeof item !== 'string' && item.settings) {
                    if (widthInput) widthInput.value = item.settings.width;
                    if (heightInput) heightInput.value = item.settings.height;
                    if (negativePromptInput) negativePromptInput.value = item.settings.negative_prompt || '';

                    if (guidanceScaleInput) {
                        guidanceScaleInput.value = item.settings.guidance_scale;
                        if (guidanceScaleValue) guidanceScaleValue.textContent = item.settings.guidance_scale;
                    }

                    if (inferenceStepsInput) {
                        inferenceStepsInput.value = item.settings.num_inference_steps;
                        if (inferenceStepsValue) inferenceStepsValue.textContent = item.settings.num_inference_steps;
                    }

                    if (modelSelector && item.settings.model) {
                        // Update UI only. Note: This does not trigger the backend model switch automatically
                        // to avoid freezing the UI while browsing history.
                        modelSelector.value = item.settings.model;
                    }
                }
            });
            historyList.appendChild(historyItem);
        });
    } catch (error) {
        console.error('Error fetching history:', error);
        if (historyList) {
            historyList.innerHTML = '<p>Could not load history.</p>';
        }
    }
};

// Function to update clear button visibility
const updateClearButtonVisibility = () => {
    if (clearImageButton) {
        if (initImageInput && initImageInput.files && initImageInput.files.length > 0) {
            clearImageButton.style.display = 'inline-block';
        } else {
            clearImageButton.style.display = 'none';
        }
    }
};

if (generateButton && cancelButton && downloadButton && promptInput && widthInput && heightInput && imageDisplay && initImageInput && clearImageButton && historyList && generateNewButton && deleteHistoryButton && negativePromptInput && guidanceScaleInput && guidanceScaleValue && inferenceStepsInput && inferenceStepsValue && progressContainer && progressBar && progressText && modelSelector) {
    // Initial fetch and render history
    fetchAndRenderHistory();

    // Initial visibility check for clear button
    updateClearButtonVisibility();

    // Event listener for file input change to show/hide clear button
    initImageInput.addEventListener('change', updateClearButtonVisibility);

    // --- Model Manager Logic ---
    const fetchModels = async () => {
        try {
            const response = await fetch(`${BACKEND_URL}/api/models`);
            const data = await response.json();

            modelSelector.innerHTML = '';

            data.models.forEach((modelName: string) => {
                const option = document.createElement('option');
                option.value = modelName;
                option.textContent = modelName;
                modelSelector.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to fetch models:', error);
        }
    };

    modelSelector.addEventListener('change', async (e) => {
        const selectedModel = (e.target as HTMLSelectElement).value;

        if (loadingOverlay) loadingOverlay.style.display = 'flex';

        try {
            const response = await fetch(`${BACKEND_URL}/api/models/switch`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_name: selectedModel }),
            });

            if (!response.ok) throw new Error('Failed to switch model');
            console.log(`Switched to ${selectedModel}`);
        } catch (error) {
            console.error(error);
            alert('Error switching model. Check console.');
        } finally {
            if (loadingOverlay) loadingOverlay.style.display = 'none';
        }
    });

    fetchModels();

    // Event listener for Clear Image Button
    clearImageButton.addEventListener('click', () => {
        initImageInput.value = ''; // Clear the file input
        updateClearButtonVisibility(); // Update button visibility
    });

    // Event listeners for sliders
    guidanceScaleInput.addEventListener('input', () => {
        guidanceScaleValue.textContent = guidanceScaleInput.value;
    });

    inferenceStepsInput.addEventListener('input', () => {
        inferenceStepsValue.textContent = inferenceStepsInput.value;
    });

    // Event listener for Delete History Button
    deleteHistoryButton.addEventListener('click', async () => {
        if (confirm('Are you sure you want to delete all history? This cannot be undone.')) {
            try {
                const response = await fetch(`${BACKEND_URL}/api/history`, { method: 'DELETE' });
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                fetchAndRenderHistory(); // Refresh history list
                imageDisplay.innerHTML = '<p>Generated image will appear here.</p>'; // Clear main display
                downloadButton.style.display = 'none'; // Hide download button
            } catch (error) {
                console.error('Error deleting history:', error);
                alert('Failed to delete history.');
            }
        }
    });

    // Event listener for New Generate Button
    generateNewButton.addEventListener('click', async () => {
        window.location.reload();
    });
    // Event listener for Generate Button
    generateButton.addEventListener('click', async () => {
        const prompt = promptInput.value;
        const width = parseInt(widthInput.value, 10);
        const height = parseInt(heightInput.value, 10);
        const initImage = initImageInput.files ? initImageInput.files[0] : null;
        const negativePrompt = negativePromptInput.value;
        const guidanceScale = parseFloat(guidanceScaleInput.value);
        const inferenceSteps = parseInt(inferenceStepsInput.value, 10);

        if (!prompt) {
            alert('Please enter a prompt!');
            return;
        }
        if (isNaN(width) || isNaN(height) || width <= 0 || height <= 0) {
            alert('Please enter valid width and height values!');
            return;
        }

        imageDisplay.innerHTML = '<p>Generating image...</p>';
        downloadButton.style.display = 'none'; // Hide download button during generation
        cancelButton.style.display = 'inline-block'; // Show cancel button
        lastImageData = null;

        // Show progress bar
        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';
        progressText.textContent = '0%';

        // Prepare data for WebSocket
        const requestData: any = {
            prompt: prompt,
            width: width,
            height: height,
            negative_prompt: negativePrompt,
            guidance_scale: guidanceScale,
            num_inference_steps: inferenceSteps
        };

        // Helper to read file as base64
        const readFileAsBase64 = (file: File): Promise<string> => {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result as string);
                reader.onerror = reject;
                reader.readAsDataURL(file);
            });
        };

        if (initImage) {
            try {
                requestData.init_image = await readFileAsBase64(initImage);
            } catch (e) {
                console.error("Error reading init image", e);
                alert("Failed to read initial image.");
                return;
            }
        }

        // Initialize WebSocket
        // Connect directly to backend port 8000 to bypass potential Vite proxy issues with WebSockets
        const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws/generate`);

        // Handle Cancel Button Click
        const handleCancel = () => {
            ws.close();
            imageDisplay.innerHTML = '<p>Generation cancelled.</p>';
        };
        cancelButton.onclick = handleCancel;

        ws.onopen = () => {
            console.log('WebSocket connected');
            ws.send(JSON.stringify(requestData));
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.status === 'progress') {
                const percentage = data.percentage;
                progressBar.style.width = `${percentage}%`;
                progressText.textContent = `${percentage}% (Step ${data.step}/${data.total_steps})`;
            } else if (data.status === 'completed') {
                if (data.image) {
                    lastImageData = `data:image/png;base64,${data.image}`;
                    imageDisplay.innerHTML = `<img src="${lastImageData}" alt="Generated Image" style="width:${width}px; height:${height}px;">`;
                    downloadButton.style.display = 'inline-block';
                    fetchAndRenderHistory();
                }
                progressContainer.style.display = 'none';
                ws.close();
            } else if (data.status === 'error') {
                imageDisplay.innerHTML = `<p>Error: ${data.message}</p>`;
                progressContainer.style.display = 'none';
                ws.close();
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            imageDisplay.innerHTML = '<p>WebSocket error occurred.</p>';
            progressContainer.style.display = 'none';
        };

        ws.onclose = () => {
            console.log('WebSocket closed');
            cancelButton.style.display = 'none'; // Hide cancel button
            progressContainer.style.display = 'none'; // Ensure progress bar is hidden
            cancelButton.onclick = null; // Cleanup event listener
        };
    });

    // Event listener for Download Button
    downloadButton.addEventListener('click', () => {
        if (lastImageData) {
            const a = document.createElement('a');
            a.href = lastImageData;

            const safeFilename = promptInput.value
                .replace(/[^a-z0-9\s-]/gi, '')
                .trim()
                .replace(/\s+/g, '-')
                .substring(0, 60)
                .toLowerCase() || 'generated-image';

            a.download = `${safeFilename}.png`;
            a.click();
        }
    });

    // Event listener for Enter key in prompt input
    promptInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault(); // Prevent new line in textarea
            generateButton.click(); // Trigger generate button click
        }
    });

    // Global event listener for Escape key
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            if (abortController) {
                abortController.abort(); // Abort ongoing fetch request
                console.log('Generation process interrupted by Escape key.');
                imageDisplay.innerHTML = '<p>Image generation aborted.</p>'; // Update UI on abort
            }
        }
    });
}
