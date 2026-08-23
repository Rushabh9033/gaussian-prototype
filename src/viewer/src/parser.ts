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
    manifest: Manifest;
    splatsData: Float32Array; // N * 9 floats
    previewUrl?: string;
    packageSize: number;
}

export async function parsePackage(file: File): Promise<PackageData> {
    const zip = await JSZip.loadAsync(file);
    
    const manifestFile = zip.file("manifest.json");
    if (!manifestFile) throw new Error("manifest.json not found in package");
    const manifestStr = await manifestFile.async("string");
    const manifest = JSON.parse(manifestStr) as Manifest;
    
    const splatsFile = zip.file("splats.bin");
    if (!splatsFile) throw new Error("splats.bin not found in package");
    
    const splatsBuffer = await splatsFile.async("arraybuffer");
    const splatsData = new Float32Array(splatsBuffer);
    
    if (splatsData.length !== manifest.gaussian_count * 9) {
        throw new Error("splats.bin size does not match gaussian_count * 9 floats");
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
        previewUrl,
        packageSize: file.size
    };
}
