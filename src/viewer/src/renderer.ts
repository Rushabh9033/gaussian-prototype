export class GaussianRenderer {
    private canvas: HTMLCanvasElement;
    private gl: WebGL2RenderingContext;
    private program: WebGLProgram | null = null;
    
    private vao: WebGLVertexArrayObject | null = null;
    private instanceBuffer: WebGLBuffer | null = null;
    private quadBuffer: WebGLBuffer | null = null;
    
    private numInstances = 0;
    
    private panX = 0;
    private panY = 0;
    private zoom = 1.0;
    private encodedWidth = 1.0;
    private encodedHeight = 1.0;
    
    // Drag state
    private isDragging = false;
    private lastMouseX = 0;
    private lastMouseY = 0;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        const gl = canvas.getContext('webgl2', { alpha: false, premultipliedAlpha: false });
        if (!gl) throw new Error("WebGL2 not supported");
        this.gl = gl;
        
        this.initWebGL();
        this.setupEvents();
    }
    
    private initWebGL() {
        const gl = this.gl;
        
        const vsSource = `#version 300 es
        precision highp float;
        
        layout(location = 0) in vec2 a_quad;
        layout(location = 1) in vec2 i_pos;
        layout(location = 2) in vec2 i_scale;
        layout(location = 3) in float i_rot;
        layout(location = 4) in vec3 i_color;
        layout(location = 5) in float i_alpha;
        
        uniform mat3 u_transform;
        
        out vec2 v_quad;
        out vec3 v_color;
        out float v_alpha;
        
        void main() {
            v_quad = a_quad;
            v_color = i_color;
            v_alpha = i_alpha;
            
            float c = cos(i_rot);
            float s = sin(i_rot);
            mat2 rotMat = mat2(c, s, -s, c);
            
            vec2 scaled = a_quad * i_scale;
            vec2 local_pos = rotMat * scaled;
            vec2 abs_pos = local_pos + i_pos;
            
            vec3 transformed = u_transform * vec3(abs_pos, 1.0);
            gl_Position = vec4(transformed.xy, 0.0, 1.0);
        }
        `;
        
        const fsSource = `#version 300 es
        precision highp float;
        
        in vec2 v_quad;
        in vec3 v_color;
        in float v_alpha;
        
        out vec4 fragColor;
        
        void main() {
            float power = -0.5 * dot(v_quad, v_quad);
            float a = v_alpha * exp(power);
            if (a < 0.001) discard;
            fragColor = vec4(v_color, a);
        }
        `;
        
        const vs = this.compileShader(gl.VERTEX_SHADER, vsSource);
        const fs = this.compileShader(gl.FRAGMENT_SHADER, fsSource);
        
        this.program = gl.createProgram();
        gl.attachShader(this.program!, vs);
        gl.attachShader(this.program!, fs);
        gl.linkProgram(this.program!);
        
        if (!gl.getProgramParameter(this.program!, gl.LINK_STATUS)) {
            throw new Error(gl.getProgramInfoLog(this.program!) || "Link error");
        }
        
        // Quad vertices for 3 std devs
        const r = 3.0;
        const quadVertices = new Float32Array([
            -r, -r,
             r, -r,
            -r,  r,
             r,  r
        ]);
        
        this.vao = gl.createVertexArray();
        gl.bindVertexArray(this.vao);
        
        this.quadBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, quadVertices, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
        
        this.instanceBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
        
        const stride = 9 * 4;
        // i_pos
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 2, gl.FLOAT, false, stride, 0);
        gl.vertexAttribDivisor(1, 1);
        
        // i_scale
        gl.enableVertexAttribArray(2);
        gl.vertexAttribPointer(2, 2, gl.FLOAT, false, stride, 2 * 4);
        gl.vertexAttribDivisor(2, 1);
        
        // i_rot
        gl.enableVertexAttribArray(3);
        gl.vertexAttribPointer(3, 1, gl.FLOAT, false, stride, 4 * 4);
        gl.vertexAttribDivisor(3, 1);
        
        // i_color
        gl.enableVertexAttribArray(4);
        gl.vertexAttribPointer(4, 3, gl.FLOAT, false, stride, 5 * 4);
        gl.vertexAttribDivisor(4, 1);
        
        // i_alpha
        gl.enableVertexAttribArray(5);
        gl.vertexAttribPointer(5, 1, gl.FLOAT, false, stride, 8 * 4);
        gl.vertexAttribDivisor(5, 1);
        
        gl.bindVertexArray(null);
        
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.clearColor(0, 0, 0, 1);
    }
    
    private compileShader(type: number, source: string): WebGLShader {
        const gl = this.gl;
        const shader = gl.createShader(type)!;
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            throw new Error(gl.getShaderInfoLog(shader) || "Compile error");
        }
        return shader;
    }
    
    public loadSplats(data: Float32Array, width: number, height: number, preserveView = false) {
        this.encodedWidth = width;
        this.encodedHeight = height;
        
        this.numInstances = data.length / 9;
        
        // Reverse array to draw back-to-front (since PyTorch 0 was front)
        const reversed = new Float32Array(data.length);
        for (let i = 0; i < this.numInstances; i++) {
            const srcIdx = (this.numInstances - 1 - i) * 9;
            const dstIdx = i * 9;
            for (let j = 0; j < 9; j++) reversed[dstIdx + j] = data[srcIdx + j];
        }
        
        const gl = this.gl;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, reversed, gl.STATIC_DRAW);
        
        if (!preserveView) {
            this.fitToScreen();
        } else {
            this.render();
        }
    }
    
    public fitToScreen() {
        if (!this.canvas) return;
        const cw = this.canvas.width;
        const ch = this.canvas.height;
        
        const scaleX = cw / this.encodedWidth;
        const scaleY = ch / this.encodedHeight;
        this.zoom = Math.min(scaleX, scaleY) * 0.95;
        
        this.panX = (cw - this.encodedWidth * this.zoom) / 2.0;
        this.panY = (ch - this.encodedHeight * this.zoom) / 2.0;
        
        this.render();
    }
    
    private setupEvents() {
        const updateOnResize = () => {
            const parent = this.canvas.parentElement;
            if (parent) {
                this.canvas.width = parent.clientWidth;
                this.canvas.height = parent.clientHeight;
                this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
                this.render();
            }
        };
        window.addEventListener('resize', updateOnResize);
        setTimeout(updateOnResize, 0); // initial sizing
        
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const factor = Math.pow(1.001, -e.deltaY);
            
            // Zoom around mouse pointer
            const mouseX = e.offsetX;
            const mouseY = e.offsetY;
            
            this.panX = mouseX - (mouseX - this.panX) * factor;
            this.panY = mouseY - (mouseY - this.panY) * factor;
            this.zoom *= factor;
            
            this.render();
        });
        
        this.canvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.lastMouseX = e.offsetX;
            this.lastMouseY = e.offsetY;
        });
        
        window.addEventListener('mouseup', () => {
            this.isDragging = false;
        });
        
        window.addEventListener('mousemove', (e) => {
            if (this.isDragging) {
                const dx = e.offsetX - this.lastMouseX;
                const dy = e.offsetY - this.lastMouseY;
                this.panX += dx;
                this.panY += dy;
                this.lastMouseX = e.offsetX;
                this.lastMouseY = e.offsetY;
                this.render();
            }
        });
    }
    
    public getZoomPercentage(): number {
        return Math.round(this.zoom * 100);
    }
    
    public render() {
        if (!this.program || this.numInstances === 0) return;
        
        const gl = this.gl;
        gl.clear(gl.COLOR_BUFFER_BIT);
        
        gl.useProgram(this.program);
        gl.bindVertexArray(this.vao);
        
        const cw = this.canvas.width;
        const ch = this.canvas.height;
        
        // Map [0, encodedW] to screen pixels, then to clip space [-1, 1]
        // matrix translates by panX/panY, scales by zoom, then standard orthographic
        const sx = 2.0 / cw;
        const sy = -2.0 / ch; // flip Y
        
        // transformation matrix in column-major
        // WebGL clip space: [-1, 1], top-left is (-1, 1) if we flip Y
        const transform = new Float32Array([
            this.zoom * sx, 0, 0,
            0, this.zoom * sy, 0,
            this.panX * sx - 1.0, this.panY * sy + 1.0, 1.0
        ]);
        
        const loc = gl.getUniformLocation(this.program, "u_transform");
        gl.uniformMatrix3fv(loc, false, transform);
        
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.numInstances);
        
        window.dispatchEvent(new CustomEvent('render-updated'));
    }
}
