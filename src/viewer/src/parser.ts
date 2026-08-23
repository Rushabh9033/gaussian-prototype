import JSZip from 'jszip';

export interface Manifest {
    format_version: string;
    original_width: number;
    original_height: number;
    encoded_width: number;
    encoded_height: number;
    gaussian_count: number;
    binary_layout: string;
    coordinate_system: string;
    color_space: string;
    creation_timestamp: string;
    encoder_version: string;
    source_sha256: string;
    deterministic_seed: number;
    ai_generated_detail: boolean;
}

export interface PackageData {
    manifest: any;
    splatsData: Float32Array; // Base layer (or full if V1)
    splatsDataDetail?: Float32Array; // Detail layer if V2
    previewUrl?: string;
    packageSize: number;
    baseSize?: number;
    detailSize?: number;
}

function dequantize(buffer: ArrayBuffer, manifest: any): Float32Array {
    const W = manifest.quantization_rules.position.x_max;
    const H = manifest.quantization_rules.position.y_max;
    const minLogX = manifest.quantization_rules.scale.min_log_x;
    const maxLogX = manifest.quantization_rules.scale.max_log_x;
    const minLogY = manifest.quantization_rules.scale.min_log_y;
    const maxLogY = manifest.quantization_rules.scale.max_log_y;
    
    const view = new DataView(buffer);
    const N = buffer.byteLength / 14;
    const out = new Float32Array(N * 9);
    
    for (let i = 0; i < N; i++) {
        const off = i * 14;
        out[i*9 + 0] = view.getUint16(off + 0, true) / 65535.0 * W;
        out[i*9 + 1] = view.getUint16(off + 2, true) / 65535.0 * H;
        out[i*9 + 2] = Math.exp((view.getUint16(off + 4, true) / 65535.0) * (maxLogX - minLogX) + minLogX);
        out[i*9 + 3] = Math.exp((view.getUint16(off + 6, true) / 65535.0) * (maxLogY - minLogY) + minLogY);
        out[i*9 + 4] = (view.getUint16(off + 8, true) / 65535.0) * (2 * Math.PI);
        out[i*9 + 5] = view.getUint8(off + 10) / 255.0;
        out[i*9 + 6] = view.getUint8(off + 11) / 255.0;
        out[i*9 + 7] = view.getUint8(off + 12) / 255.0;
        out[i*9 + 8] = view.getUint8(off + 13) / 255.0;
    }
    return out;
}

async function sha256(buffer: ArrayBuffer) {
    const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

export async function parsePackage(file: File, onBaseLoaded?: (pkg: PackageData) => void): Promise<PackageData> {
    const zip = await JSZip.loadAsync(file);
    
    const manifestFile = zip.file("manifest.json");
    if (!manifestFile) throw new Error("manifest.json not found in package");
    const manifestStr = await manifestFile.async("string");
    const manifest = JSON.parse(manifestStr);
    
    let splatsData: Float32Array;
    let splatsDataDetail: Float32Array | undefined;
    let baseSize = 0;
    let detailSize = 0;
    
    if (manifest.format_version !== "1.0" && manifest.format_version !== "2.0") {
        throw new Error("Unsupported format version: " + manifest.format_version);
    }
    
    if (manifest.format_version === "2.0") {
        if (manifest.record_stride !== 14) throw new Error("Invalid record stride, expected 14");
        
        const baseFile = zip.file("layers/base.bin");
        if (!baseFile) throw new Error("Base layer not found");
        const baseBuf = await baseFile.async("arraybuffer");
        
        const baseLayerInfo = manifest.layers[0];
        if (baseBuf.byteLength !== baseLayerInfo.gaussian_count * 14) throw new Error("Base layer length mismatch");
        if (await sha256(baseBuf) !== baseLayerInfo.sha256) throw new Error("Base layer checksum corruption");
        
        baseSize = baseBuf.byteLength;
        splatsData = dequantize(baseBuf, manifest);
        
        if (onBaseLoaded) {
            onBaseLoaded({
                manifest,
                splatsData,
                packageSize: file.size,
                baseSize,
                detailSize: 0
            });
        }
        
        const detailFile = zip.file("layers/detail.bin");
        if (!detailFile) throw new Error("Detail layer not found");
        const detailBuf = await detailFile.async("arraybuffer");
        
        const detailLayerInfo = manifest.layers[1];
        if (detailBuf.byteLength !== detailLayerInfo.gaussian_count * 14) throw new Error("Detail layer length mismatch");
        if (await sha256(detailBuf) !== detailLayerInfo.sha256) throw new Error("Detail layer checksum corruption");
        
        if (baseLayerInfo.gaussian_count + detailLayerInfo.gaussian_count !== manifest.total_gaussian_count) {
            throw new Error("Combined gaussian count mismatch");
        }
        
        detailSize = detailBuf.byteLength;
        splatsDataDetail = dequantize(detailBuf, manifest);
    } else {
        const splatsFile = zip.file("splats.bin");
        if (!splatsFile) throw new Error("splats.bin not found in package");
        const splatsBuffer = await splatsFile.async("arraybuffer");
        if (splatsBuffer.byteLength !== manifest.gaussian_count * 36) throw new Error("splats.bin length mismatch");
        splatsData = new Float32Array(splatsBuffer);
    }
    
    const previewFile = zip.file("preview.webp") || zip.file("preview.png");
    let previewUrl: string | undefined;
    if (previewFile) {
        const blob = await previewFile.async("blob");
        previewUrl = URL.createObjectURL(blob);
    }
    
    return {
        manifest,
        splatsData,
        splatsDataDetail,
        previewUrl,
        packageSize: file.size,
        baseSize,
        detailSize
    };
}
