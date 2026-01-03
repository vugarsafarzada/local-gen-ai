const promptInput = document.querySelector<HTMLTextAreaElement>('#prompt-input');
const widthInput = document.querySelector<HTMLInputElement>('#width-input');
const heightInput = document.querySelector<HTMLInputElement>('#height-input');
const initImageInput = document.querySelector<HTMLInputElement>('#init-image'); // New file input
const clearImageButton = document.querySelector<HTMLButtonElement>('#clear-image-button'); // New clear button
const generateButton = document.querySelector<HTMLButtonElement>('#generate-button');
const downloadButton = document.querySelector<HTMLButtonElement>('#download-button');
const imageDisplay = document.querySelector<HTMLDivElement>('#image-display');

// Assuming the backend is running on http://localhost:8000
const BACKEND_URL = '/api'; // Updated to use the Vite proxy

let lastImageData: string | null = null;
let abortController: AbortController | null = null; // To manage ongoing fetch requests

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

if (generateButton && downloadButton && promptInput && widthInput && heightInput && imageDisplay && initImageInput && clearImageButton) {
    // Initial visibility check for clear button
    updateClearButtonVisibility();

    // Event listener for file input change to show/hide clear button
    initImageInput.addEventListener('change', updateClearButtonVisibility);

    // Event listener for Clear Image Button
    clearImageButton.addEventListener('click', () => {
        initImageInput.value = ''; // Clear the file input
        updateClearButtonVisibility(); // Update button visibility
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
                // Optionally, update UI to reflect interruption
            }
        }
    });
}
