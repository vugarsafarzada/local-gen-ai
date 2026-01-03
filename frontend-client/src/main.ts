const promptInput = document.querySelector<HTMLTextAreaElement>('#prompt-input');
const widthInput = document.querySelector<HTMLInputElement>('#width-input');
const heightInput = document.querySelector<HTMLInputElement>('#height-input');
const initImageInput = document.querySelector<HTMLInputElement>('#init-image'); // New file input
const clearImageButton = document.querySelector<HTMLButtonElement>('#clear-image-button'); // New clear button
const deleteHistoryButton = document.querySelector<HTMLButtonElement>('#delete-history-button'); // New delete button
const generateButton = document.querySelector<HTMLButtonElement>('#generate-button');
const downloadButton = document.querySelector<HTMLButtonElement>('#download-button');
const imageDisplay = document.querySelector<HTMLDivElement>('#image-display');
const historyList = document.querySelector<HTMLDivElement>('#history-list'); // Changed from historyGrid to historyList

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

            if (typeof item === 'string') {
                // Handle old format
                filename = item;
                prompt = item.replace('.png', '').replace(/_/g, ' ');
            } else {
                // Handle new format
                filename = item.filename;
                prompt = item.prompt;
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
                        const deleteResponse = await fetch(`${BACKEND_URL}/api/history/item/${filename}`, { method: 'DELETE' }); // Corrected URL
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

if (generateButton && downloadButton && promptInput && widthInput && heightInput && imageDisplay && initImageInput && clearImageButton && historyList && deleteHistoryButton) {
    // Initial fetch and render history
    fetchAndRenderHistory();

    // Initial visibility check for clear button
    updateClearButtonVisibility();

    // Event listener for file input change to show/hide clear button
    initImageInput.addEventListener('change', updateClearButtonVisibility);

    // Event listener for Clear Image Button
    clearImageButton.addEventListener('click', () => {
        initImageInput.value = ''; // Clear the file input
        updateClearButtonVisibility(); // Update button visibility
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

    // Event listener for Generate Button
    generateButton.addEventListener('click', async () => {
        const prompt = promptInput.value;
        const width = parseInt(widthInput.value, 10);
        const height = parseInt(heightInput.value, 10);
        const initImage = initImageInput.files ? initImageInput.files[0] : null;

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
        lastImageData = null;

        // Abort any previous ongoing request
        if (abortController) {
            abortController.abort();
        }
        abortController = new AbortController();
        const signal = abortController.signal;

        try {
            // Construct FormData
            const formData = new FormData();
            formData.append('prompt', prompt);
            formData.append('width', String(width));
            formData.append('height', String(height));
            if (initImage) {
                formData.append('init_image', initImage);
            }

            const response = await fetch(`${BACKEND_URL}/generate`, {
                method: 'POST',
                // No 'Content-Type' header; browser sets it with boundary for FormData
                body: formData,
                signal: signal, // Pass the signal to the fetch request
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            if (data.image) {
                lastImageData = `data:image/png;base64,${data.image}`;
                imageDisplay.innerHTML = `<img src="${lastImageData}" alt="Generated Image" style="width:${width}px; height:${height}px;">`;
                downloadButton.style.display = 'inline-block'; // Show download button
                fetchAndRenderHistory(); // Refresh history
            } else {
                imageDisplay.innerHTML = '<p>Error: No image received from the backend.</p>';
            }

        } catch (error: any) { // Use 'any' for error type to safely access .name
            if (error.name === 'AbortError') {
                console.log('Image generation aborted.');
                imageDisplay.innerHTML = '<p>Image generation aborted.</p>';
            } else {
                console.error('Error generating image:', error);
                imageDisplay.innerHTML = `<p>Error generating image: ${error.message}</p>`;
            }
        } finally {
            abortController = null; // Clear controller after request completes or is aborted
        }
    });

    // Event listener for Download Button
    downloadButton.addEventListener('click', () => {
        if (lastImageData) {
            const a = document.createElement('a');
            a.href = lastImageData;
            a.download = 'generated-image.png';
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
