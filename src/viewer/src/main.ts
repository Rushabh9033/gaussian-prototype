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

const branchUi = document.getElementById('branch-ui') as HTMLDivElement;
const badgeSynthetic = document.getElementById('badge-synthetic') as HTMLSpanElement;
const valBranchLevel = document.getElementById('val-branch-level') as HTMLSpanElement;
const valBranchMax = document.getElementById('val-branch-max') as HTMLSpanElement;
const btnExplore = document.getElementById('btn-explore') as HTMLButtonElement;
const btnBackBase = document.getElementById('btn-back-base') as HTMLButtonElement;
const branchHighlight = document.getElementById('branch-highlight') as HTMLDivElement;
const branchImg = document.getElementById('branch-img') as HTMLImageElement;
const qualityRadios = document.getElementById('quality-radios') as HTMLDivElement;

let renderer: GaussianRenderer;
let currentPkg: any = null;
let currentBranch: any = null;
let branchMode = false;
let currentLevel = 0;
let levelUrls: Record<number, string> = {};

try {
    renderer = new GaussianRenderer(canvas);
} catch (e) {
    errorMsg.textContent = "Failed to initialize WebGL: " + (e as Error).message;
}

function renderCurrentLayer(preserveView = true) {
    if (!currentPkg || branchMode) return;
    
    let activeData = currentPkg.splatsData;
    let count = currentPkg.splatsData.length / 9;
    
    if (radioFull.checked && currentPkg.splatsDataDetail) {
        const fullData = new Float32Array(currentPkg.splatsData.length + currentPkg.splatsDataDetail.length);
        fullData.set(currentPkg.splatsData);
        fullData.set(currentPkg.splatsDataDetail, currentPkg.splatsData.length);
        activeData = fullData;
        count = fullData.length / 9;
    }
    
    renderer.loadSplats(activeData, currentPkg.manifest.encoded_width, currentPkg.manifest.encoded_height, preserveView);
    valCount.textContent = count.toLocaleString();
    if (currentPkg.splatsDataDetail) {
        const total = (currentPkg.splatsData.length + currentPkg.splatsDataDetail.length) / 9;
        valTotalCount.textContent = `/ ${total.toLocaleString()}`;
    } else {
        valTotalCount.textContent = "";
    }
}

function syncOverlays() {
    if (!currentPkg) return;
    
    // transform CSS logic for highlight
    if (currentBranch && !branchMode) {
        branchHighlight.style.display = "block";
        const bbox = currentBranch.root_bbox_pixels; // [x, y, w, h]
        const hx = bbox[0];
        const hy = bbox[1];
        const hw = bbox[2];
        const hh = bbox[3];
        
        // Convert to screen space
        const left = hx * renderer.zoom + renderer.panX;
        const top = hy * renderer.zoom + renderer.panY;
        const w = hw * renderer.zoom;
        const h = hh * renderer.zoom;
        
        branchHighlight.style.left = left + "px";
        branchHighlight.style.top = top + "px";
        branchHighlight.style.width = w + "px";
        branchHighlight.style.height = h + "px";
    } else {
        branchHighlight.style.display = "none";
    }
    
    if (branchMode && currentBranch) {
        branchImg.style.display = "block";
        
        // We want the image to align perfectly with the source image's coordinate space.
        // The image at level L covers a specific root_bbox.
        const levelInfo = currentBranch.levels.find((l:any) => l.level === currentLevel);
        if (levelInfo) {
            const bbox = levelInfo.root_bbox;
            const hx = bbox[0];
            const hy = bbox[1];
            const hw = bbox[2];
            const hh = bbox[3];
            
            const left = hx * renderer.zoom + renderer.panX;
            const top = hy * renderer.zoom + renderer.panY;
            const w = hw * renderer.zoom;
            const h = hh * renderer.zoom;
            
            branchImg.style.left = left + "px";
            branchImg.style.top = top + "px";
            branchImg.style.width = w + "px";
            branchImg.style.height = h + "px";
        }
    } else {
        branchImg.style.display = "none";
    }
}

async function loadLevel(lvl: number) {
    if (!currentBranch || !currentPkg.fetchBranchLevel) return;
    currentLevel = lvl;
    valBranchLevel.textContent = lvl.toString();
    
    if (!levelUrls[lvl]) {
        try {
            levelUrls[lvl] = await currentPkg.fetchBranchLevel(currentBranch.branch_id, lvl);
        } catch (e) {
            console.error(e);
            return;
        }
    }
    branchImg.src = levelUrls[lvl];
    syncOverlays();
}

btnExplore.addEventListener('click', async () => {
    if (!currentBranch) return;
    branchMode = true;
    btnExplore.style.display = "none";
    btnBackBase.style.display = "inline-block";
    badgeSynthetic.style.display = "inline-block";
    qualityRadios.style.opacity = "0.3";
    
    // Hide gaussians
    renderer.loadSplats(new Float32Array(0), currentPkg.manifest.encoded_width, currentPkg.manifest.encoded_height, true);
    
    // Zoom to branch
    const bbox = currentBranch.root_bbox_pixels;
    const cw = canvas.width;
    const ch = canvas.height;
    
    const scaleX = cw / bbox[2];
    const scaleY = ch / bbox[3];
    renderer.zoom = Math.min(scaleX, scaleY) * 0.95;
    renderer.panX = (cw - bbox[2] * renderer.zoom) / 2.0 - bbox[0] * renderer.zoom;
    renderer.panY = (ch - bbox[3] * renderer.zoom) / 2.0 - bbox[1] * renderer.zoom;
    renderer.render();
    
    await loadLevel(1);
});

btnBackBase.addEventListener('click', () => {
    branchMode = false;
    btnExplore.style.display = "inline-block";
    btnBackBase.style.display = "none";
    badgeSynthetic.style.display = "none";
    qualityRadios.style.opacity = "1";
    
    renderCurrentLayer(true);
    syncOverlays();
});

// hook into wheel to advance levels when in branch mode
window.addEventListener('wheel', () => {
    if (branchMode && currentBranch) {
        // Find optimal level based on zoom
        // Base image is zoom=1. Level 1 covers a smaller area.
        // If zoom is large enough to see level 2 clearly, switch to level 2.
        let optimalLvl = 1;
        for (const l of currentBranch.levels) {
            const lSize = l.root_bbox[2] * renderer.zoom;
            if (lSize > 128) {
                optimalLvl = l.level;
            }
        }
        if (optimalLvl !== currentLevel) {
            loadLevel(optimalLvl);
        }
    }
}, {passive: true});

radioBase.addEventListener('change', () => renderCurrentLayer(true));
radioFull.addEventListener('change', () => renderCurrentLayer(true));

fileInput.addEventListener('change', async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    
    errorMsg.textContent = "Loading package...";
    statsDiv.style.display = "none";
    radioFull.disabled = true;
    branchUi.style.display = "none";
    currentBranch = null;
    branchMode = false;
    levelUrls = {};
    
    try {
        const pkg = await parsePackage(file, (basePkg) => {
            currentPkg = basePkg;
            radioBase.checked = true;
            renderCurrentLayer(false);
            
            valDim.textContent = `${basePkg.manifest.original_width}x${basePkg.manifest.original_height}`;
            valSize.textContent = (basePkg.packageSize / 1024).toFixed(1);
            
            v2Stats.style.display = "block";
            valBaseSize.textContent = (basePkg.baseSize! / 1024).toFixed(1);
            valDetailSize.textContent = "Loading...";
            statsDiv.style.display = "block";
            
            if (basePkg.branches && basePkg.branches.length > 0) {
                currentBranch = basePkg.branches[0];
                branchUi.style.display = "block";
                valBranchMax.textContent = currentBranch.level_count.toString();
                syncOverlays();
            }
        });
        
        currentPkg = pkg;
        
        if (pkg.manifest.format_version === "2.0" || pkg.manifest.format_version === "3.0") {
            radioFull.disabled = false;
            valDetailSize.textContent = (pkg.detailSize! / 1024).toFixed(1);
        } else {
            v2Stats.style.display = "none";
            radioBase.checked = true;
            renderCurrentLayer(false);
            valDim.textContent = `${pkg.manifest.original_width}x${pkg.manifest.original_height}`;
            valSize.textContent = (pkg.packageSize / 1024).toFixed(1);
            statsDiv.style.display = "block";
        }
        
        if (pkg.branches && pkg.branches.length > 0) {
            currentBranch = pkg.branches[0];
            branchUi.style.display = "block";
            valBranchMax.textContent = currentBranch.level_count.toString();
            syncOverlays();
        }
        errorMsg.textContent = "";
        
    } catch (err) {
        console.error(err);
        errorMsg.textContent = "Error: " + (err as Error).message;
    }
});

btnFit.addEventListener('click', () => {
    renderer.fitToScreen();
    syncOverlays();
});

window.addEventListener('render-updated', () => {
    valZoom.textContent = renderer.getZoomPercentage().toString();
    syncOverlays();
});

