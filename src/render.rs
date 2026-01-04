use wasm_bindgen::JsCast;
use web_sys::{
    window, HtmlCanvasElement, WebGl2RenderingContext, WebGlBuffer, WebGlProgram, WebGlShader,
    WebGlUniformLocation,
};

const BASELINE_TOP: f32 = 0.10;
const BASELINE_BOTTOM: f32 = 0.98;
const AMPLITUDE_SCALE: f32 = 10.0;

const CENTER_BOOST: f32 = 0.8;
const EDGE_DIP: f32 = 0.8;
const CENTER_SIGMA: f32 = 0.12;

#[derive(Debug, Clone)]
pub struct RenderBand {
    /// (0 = lowest frequency band).
    pub index: usize,
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct RenderFrame {
    pub bands: Vec<RenderBand>,
}

pub struct Renderer {
    canvas: HtmlCanvasElement,
    context: WebGl2RenderingContext,
    program: WebGlProgram,
    position_buffer: WebGlBuffer,
    uniforms: Uniforms,
    attributes: Attributes,
}

impl Renderer {
    pub fn new() -> Result<Renderer, String> {
        let document = window()
            .ok_or("No global window exists")?
            .document()
            .ok_or("Failed to get document")?;
        let canvas = document
            .get_element_by_id("canvas")
            .ok_or("Failed to get document element 'canvas'")?
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .map_err(|_| "Failed to convert to HtmlCanvasElement")?;

        let canvas_element: web_sys::Element = canvas
            .clone()
            .dyn_into()
            .map_err(|_| "Failed to convert to Element")?;
        let width = canvas_element.client_width();
        let height = canvas_element.client_height();
        let width = width.max(1) as u32;
        let height = height.max(1) as u32;
        canvas.set_width(width);
        canvas.set_height(height);

        let context = canvas
            .get_context("webgl2")
            .map_err(|_| "Failed to get WebGL2 context")?
            .ok_or("Failed to get WebGL2 context")?
            .dyn_into::<WebGl2RenderingContext>()
            .map_err(|_| "Failed to convert to WebGl2RenderingContext")?;

        context.viewport(0, 0, width as i32, height as i32);

        let vertex_shader = vertex_shader(&context)?;
        let fragment_shader = fragment_shader(&context)?;
        let program = context
            .create_program()
            .ok_or("Unable to create shader program")?;

        context.attach_shader(&program, &vertex_shader);
        context.attach_shader(&program, &fragment_shader);
        context.link_program(&program);

        if !context
            .get_program_parameter(&program, WebGl2RenderingContext::LINK_STATUS)
            .as_bool()
            .unwrap_or(false)
        {
            return Err(context
                .get_program_info_log(&program)
                .unwrap_or_else(|| String::from("Unknown error linking program")));
        }

        let position_buffer = setup_buffer(&context)?;

        let color = context
            .get_uniform_location(&program, "uColor")
            .ok_or("Cannot find uniform location for uColor")?;

        let vertex_position = context.get_attrib_location(&program, "aPosition") as u32;

        let uniforms = Uniforms { color };
        let attributes = Attributes { vertex_position };

        Ok(Renderer {
            canvas,
            context,
            program,
            position_buffer,
            uniforms,
            attributes,
        })
    }

    pub fn resize(&self) -> Result<(), String> {
        let element: web_sys::Element = self
            .canvas
            .clone()
            .dyn_into()
            .map_err(|_| "Failed to convert canvas to Element")?;
        let width = element.client_width().max(1) as u32;
        let height = element.client_height().max(1) as u32;

        self.canvas.set_width(width);
        self.canvas.set_height(height);
        self.context.viewport(0, 0, width as i32, height as i32);
        Ok(())
    }

    pub fn render(&self, frame: &RenderFrame) -> Result<(), String> {
        let context = &self.context;

        context.clear_color(0.0, 0.0, 0.0, 1.0);
        context.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT);

        context.use_program(Some(&self.program));
        context.bind_buffer(
            WebGl2RenderingContext::ARRAY_BUFFER,
            Some(&self.position_buffer),
        );

        context.vertex_attrib_pointer_with_i32(
            self.attributes.vertex_position,
            2,
            WebGl2RenderingContext::FLOAT,
            false,
            0,
            0,
        );
        context.enable_vertex_attrib_array(self.attributes.vertex_position);

        let width = context.drawing_buffer_width() as f32;
        let height = context.drawing_buffer_height() as f32;

        let band_count = frame.bands.len();
        let spacing = if band_count > 1 {
            (BASELINE_BOTTOM - BASELINE_TOP) / (band_count - 1) as f32
        } else {
            0.0
        };
        let amplitude = if spacing > 0.0 {
            spacing * AMPLITUDE_SCALE
        } else {
            0.25
        };

        let mut band_refs: Vec<(f32, &RenderBand)> = frame
            .bands
            .iter()
            .map(|band| {
                let clamped_index = band.index.min(band_count.saturating_sub(1));
                let visual_index = band_count
                    .saturating_sub(1)
                    .saturating_sub(clamped_index);
                let baseline = if band_count > 1 {
                    BASELINE_TOP + spacing * visual_index as f32
                } else {
                    0.5
                };
                (baseline.clamp(0.0, 1.0), band)
            })
            .collect();
        band_refs.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for (baseline, band) in band_refs {
            let curve_points = build_curve_points(&band.samples, baseline, amplitude);
            if curve_points.len() < 2 {
                continue;
            }

            let fill_vertices = build_band_fill_vertices(&curve_points, baseline);
            draw_vertices(
                context,
                &self.uniforms,
                &fill_vertices,
                [0.0, 0.0, 0.0, 1.0],
                WebGl2RenderingContext::TRIANGLE_STRIP,
            );

            let line_vertices = build_thick_line_vertices(
                &curve_points,
                2.0,
                width,
                height,
            );
            draw_vertices(
                context,
                &self.uniforms,
                &line_vertices,
                [1.0, 1.0, 1.0, 1.0],
                WebGl2RenderingContext::TRIANGLE_STRIP,
            );
        }

        Ok(())
    }
}

fn draw_vertices(
    context: &WebGl2RenderingContext,
    uniforms: &Uniforms,
    vertices: &[f32],
    color: [f32; 4],
    mode: u32,
) {
    if vertices.is_empty() {
        return;
    }

    unsafe {
        let vertices_array = js_sys::Float32Array::view(vertices);
        context.buffer_data_with_array_buffer_view(
            WebGl2RenderingContext::ARRAY_BUFFER,
            &vertices_array,
            WebGl2RenderingContext::DYNAMIC_DRAW,
        );
    }

    context.uniform4f(
        Some(&uniforms.color),
        color[0],
        color[1],
        color[2],
        color[3],
    );

    let vertex_count = (vertices.len() / 2) as i32;
    context.draw_arrays(mode, 0, vertex_count);
}

fn build_curve_points(samples: &[f32], baseline: f32, amplitude: f32) -> Vec<[f32; 2]> {
    if samples.len() < 2 {
        return Vec::new();
    }

    let count = samples.len();
    let denom = (count - 1) as f32;
    let mut points = Vec::with_capacity(count);

    for (i, sample) in samples.iter().enumerate() {
        let x = i as f32 / denom;
        let gain = center_envelope(x);
        let curve = (baseline - sample * amplitude * gain).clamp(0.0, 1.0);
        points.push([x, curve]);
    }

    points
}

fn center_envelope(x: f32) -> f32 {
    let dx = x - 0.5;
    let sigma2 = CENTER_SIGMA * CENTER_SIGMA;
    let gaussian = (-dx * dx / (2.0 * sigma2)).exp();
    (1.0 - EDGE_DIP) + (CENTER_BOOST + EDGE_DIP) * gaussian
}

fn build_band_fill_vertices(curve_points: &[[f32; 2]], baseline: f32) -> Vec<f32> {
    let mut vertices = Vec::with_capacity(curve_points.len() * 4);
    for point in curve_points {
        vertices.push(point[0]);
        vertices.push(baseline);
        vertices.push(point[0]);
        vertices.push(point[1]);
    }
    vertices
}

fn build_thick_line_vertices(
    curve_points: &[[f32; 2]],
    thickness_px: f32,
    width: f32,
    height: f32,
) -> Vec<f32> {
    if curve_points.len() < 2 || width <= 0.0 || height <= 0.0 {
        return Vec::new();
    }

    let half = thickness_px * 0.5;
    let mut points_px = Vec::with_capacity(curve_points.len());
    for point in curve_points {
        points_px.push([point[0] * width, point[1] * height]);
    }

    let mut vertices = Vec::with_capacity(curve_points.len() * 4);
    for i in 0..points_px.len() {
        let prev = if i == 0 { points_px[i] } else { points_px[i - 1] };
        let next = if i + 1 == points_px.len() {
            points_px[i]
        } else {
            points_px[i + 1]
        };

        let dx = next[0] - prev[0];
        let dy = next[1] - prev[1];
        let len = (dx * dx + dy * dy).sqrt();
        let (nx, ny) = if len > 0.0 {
            let tx = dx / len;
            let ty = dy / len;
            (-ty, tx)
        } else {
            (0.0, -1.0)
        };

        let offset_x = nx * half;
        let offset_y = ny * half;
        let p = points_px[i];

        let top = [(p[0] + offset_x) / width, (p[1] + offset_y) / height];
        let bottom = [(p[0] - offset_x) / width, (p[1] - offset_y) / height];

        vertices.push(top[0]);
        vertices.push(top[1]);
        vertices.push(bottom[0]);
        vertices.push(bottom[1]);
    }

    vertices
}

fn compile_shader(
    context: &WebGl2RenderingContext,
    shader_type: u32,
    source: &str,
) -> Result<WebGlShader, String> {
    let shader = context
        .create_shader(shader_type)
        .ok_or_else(|| String::from("Unable to create shader object"))?;
    context.shader_source(&shader, source);
    context.compile_shader(&shader);

    if context
        .get_shader_parameter(&shader, WebGl2RenderingContext::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(shader)
    } else {
        Err(context
            .get_shader_info_log(&shader)
            .unwrap_or_else(|| String::from("Unknown error creating shader")))
    }
}

fn vertex_shader(context: &WebGl2RenderingContext) -> Result<WebGlShader, String> {
    compile_shader(
        &context,
        WebGl2RenderingContext::VERTEX_SHADER,
        r#"
        attribute vec2 aPosition;
        void main() {
            vec2 clip = vec2(aPosition.x * 2.0 - 1.0, 1.0 - aPosition.y * 2.0);
            gl_Position = vec4(clip, 0.0, 1.0);
        }
        "#,
    )
}

fn fragment_shader(context: &WebGl2RenderingContext) -> Result<WebGlShader, String> {
    compile_shader(
        &context,
        WebGl2RenderingContext::FRAGMENT_SHADER,
        r#"
        precision mediump float;
        uniform vec4 uColor;
        void main() {
            gl_FragColor = uColor;
        }
        "#,
    )
}

struct Uniforms {
    color: WebGlUniformLocation,
}

struct Attributes {
    vertex_position: u32,
}

fn setup_buffer(context: &WebGl2RenderingContext) -> Result<WebGlBuffer, String> {
    let position_buffer = context
        .create_buffer()
        .ok_or("Failed to create buffer")?;
    context.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&position_buffer));
    Ok(position_buffer)
}
