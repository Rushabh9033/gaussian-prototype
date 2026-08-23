import { parsePackage } from './parser';
import { GaussianRenderer } from './renderer';

const fileInput = document.getElementById('file-input') as HTMLInputElement;
const canvas = document.getElementById('viewer-canvas') as HTMLCanvasElement;
const errorMsg = document.getElementById('error-msg') as HTMLDivElement;
const statsDiv = document.getElementById('stats') as HTMLDivElement;

const valZoom = document.getElementById('val-zoom') as HTMLSpanElement;
const valDim = document.getElementById('val-dim') as HTMLSpanElement;
const valCount = document.getElementById('val-count') as HTMLSpanElement;
const valSize = document.getElementById('val-size') as HTMLSpanElement;
const btnFit = document.getElementById('btn-fit') as HTMLButtonElement;

let renderer: GaussianRenderer;

try {
    renderer = new GaussianRenderer(canvas);
} catch (e) {
    errorMsg.textContent = "Failed to initialize WebGL: " + (e as Error).message;
}

fileInput.addEventListener('change', async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    
    errorMsg.textContent = "Loading package...";
    statsDiv.style.display = "none";
    
    try {
        const pkg = await parsePackage(file);
        
        renderer.loadSplats(pkg.splatsData, pkg.manifest.encoded_width, pkg.manifest.encoded_height);
        
        valDim.textContent = `${pkg.manifest.original_width}x${pkg.manifest.original_height}`;
        valCount.textContent = pkg.manifest.gaussian_count.toLocaleString();
        valSize.textContent = (pkg.packageSize / 1024).toFixed(1);
        
        statsDiv.style.display = "block";
        errorMsg.textContent = "";
        
    } catch (err) {
        console.error(err);
        errorMsg.textContent = "Error: " + (err as Error).message;
    }
});

btnFit.addEventListener('click', () => {
    renderer.fitToScreen();
});

window.addEventListener('render-updated', () => {
    valZoom.textContent = renderer.getZoomPercentage().toString();
});
