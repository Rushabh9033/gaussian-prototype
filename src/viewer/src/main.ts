import { parsePackage } from './parser';
import { GaussianRenderer } from './renderer';

const fileInput = document.getElementById('file-input') as HTMLInputElement;
const canvas = document.getElementById('viewer-canvas') as HTMLCanvasElement;
const errorMsg = document.getElementById('error-msg') as HTMLDivElement;
const statsDiv = document.getElementById('stats') as HTMLDivElement;

const valZoom = document.getElementById('val-zoom') as HTMLSpanElement;
const valDim = document.getElementById('val-dim') as HTMLSpanElement;
const valCount = document.getElementById('val-count') as HTMLSpanElement;
const valTotalCount = document.getElementById('val-total-count') as HTMLSpanElement;
const valSize = document.getElementById('val-size') as HTMLSpanElement;
const btnFit = document.getElementById('btn-fit') as HTMLButtonElement;

const v2Stats = document.getElementById('v2-stats') as HTMLDivElement;
const valBaseSize = document.getElementById('val-base-size') as HTMLSpanElement;
const valDetailSize = document.getElementById('val-detail-size') as HTMLSpanElement;
const radioBase = document.getElementById('radio-base') as HTMLInputElement;
const radioFull = document.getElementById('radio-full') as HTMLInputElement;

let renderer: GaussianRenderer;
let currentPkg: any = null;

try {
    renderer = new GaussianRenderer(canvas);
} catch (e) {
    errorMsg.textContent = "Failed to initialize WebGL: " + (e as Error).message;
}

function renderCurrentLayer() {
    if (!currentPkg) return;
    
    let activeData = currentPkg.splatsData;
    let count = currentPkg.splatsData.length / 9;
    
    if (radioFull.checked && currentPkg.splatsDataDetail) {
        const fullData = new Float32Array(currentPkg.splatsData.length + currentPkg.splatsDataDetail.length);
        fullData.set(currentPkg.splatsData);
        fullData.set(currentPkg.splatsDataDetail, currentPkg.splatsData.length);
        activeData = fullData;
        count = fullData.length / 9;
    }
    
    renderer.loadSplats(activeData, currentPkg.manifest.encoded_width, currentPkg.manifest.encoded_height);
    valCount.textContent = count.toLocaleString();
    if (currentPkg.splatsDataDetail) {
        const total = (currentPkg.splatsData.length + currentPkg.splatsDataDetail.length) / 9;
        valTotalCount.textContent = `/ ${total.toLocaleString()}`;
    } else {
        valTotalCount.textContent = "";
    }
}

radioBase.addEventListener('change', renderCurrentLayer);
radioFull.addEventListener('change', renderCurrentLayer);

fileInput.addEventListener('change', async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    
    errorMsg.textContent = "Loading package...";
    statsDiv.style.display = "none";
    
    try {
        const pkg = await parsePackage(file);
        currentPkg = pkg;
        
        radioFull.checked = true;
        renderCurrentLayer();
        
        valDim.textContent = `${pkg.manifest.original_width}x${pkg.manifest.original_height}`;
        valSize.textContent = (pkg.packageSize / 1024).toFixed(1);
        
        if (pkg.manifest.format_version === "2.0") {
            v2Stats.style.display = "block";
            valBaseSize.textContent = (pkg.baseSize! / 1024).toFixed(1);
            valDetailSize.textContent = (pkg.detailSize! / 1024).toFixed(1);
        } else {
            v2Stats.style.display = "none";
        }
        
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
