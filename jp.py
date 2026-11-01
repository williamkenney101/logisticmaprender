import pygame
import numpy as np
from numba import jit, prange
import math
import time

# configuration
WIDTH, HEIGHT = 1200, 900
MARGIN_LEFT, MARGIN_BOTTOM, MARGIN_TOP, MARGIN_RIGHT = 85, 60, 20, 25

PLOT_W = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
PLOT_H = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Logistic Map Render")
clock = pygame.time.Clock()

# fonting
font_small = pygame.font.SysFont("Consolas", 12)
font_axis = pygame.font.SysFont("Consolas", 13, bold=True)
font_ui = pygame.font.SysFont("Consolas", 14, bold=True)

UNIVERSAL_R_MIN, UNIVERSAL_R_MAX = np.float64(2.0), np.float64(4.0) # bounds
UNIVERSAL_X_MIN, UNIVERSAL_X_MAX = np.float64(0.0), np.float64(1.0)

rmin, rmax = UNIVERSAL_R_MIN, UNIVERSAL_R_MAX
xmin, xmax = UNIVERSAL_X_MIN, UNIVERSAL_X_MAX

target_rmin, target_rmax = rmin, rmax
target_xmin, target_xmax = xmin, xmax

# resolution
BASE_STEPS = 3000       # Horizontal sampling resolution along r
BASE_SETTLE = 350       # Settle iterations
BASE_PLOT = 500         # Density sample iterations

# animation parameters
tween_speed = 0.30
EPSILON = 1e-10 # you know I used to barely understand epsilon delta proofs but this project really helped

# UI controls
show_ui = True
gamma_curve = 0.35 
drag_mode = None

# chroma effect for my project (dont pay much attention)
chroma_active = False
chroma_blend = 0.0 
chroma_time = 0.0
BLEND_SPEED = 0.02

@jit(nopython=True, fastmath=True)
def render_bifurcation_fast(density_grid, rmin, rmax, xmin, xmax, steps, settle_iters, plot_iters):
    density_grid.fill(0.0)
    
    dr = (rmax - rmin) / np.float64(steps)
    starting_x = np.float64(0.5)
    
    r_range = rmax - rmin
    x_range = xmax - xmin

    w_limit = float(PLOT_W - 1)
    h_limit = float(PLOT_H - 1)

    for step in range(steps):
        r = rmin + (np.float64(step) * dr)
        x = starting_x
        
        for _ in range(settle_iters):
            x = r * x * (1.0 - x)
            
        for _ in range(plot_iters):
            x = r * x * (1.0 - x)
            
            if xmin <= x <= xmax:
                norm_r = (r - rmin) / r_range
                norm_x = (x - xmin) / x_range
                
                fx = norm_r * w_limit
                fy = (1.0 - norm_x) * h_limit
                
                ix = int(fx)
                iy = int(fy)
                
                if 0 <= ix < PLOT_W - 1 and 0 <= iy < PLOT_H - 1:
                    dx = fx - float(ix)
                    dy = fy - float(iy)
                    
                    # 4 point bilinear
                    density_grid[ix, iy]         += (1.0 - dx) * (1.0 - dy)
                    density_grid[ix + 1, iy]     += dx * (1.0 - dy)
                    density_grid[ix, iy + 1]     += (1.0 - dx) * dy
                    density_grid[ix + 1, iy + 1] += dx * dy

@jit(nopython=True, fastmath=True)
def hsv_to_rgb(h, s, v):
   # there is probably a much better way to do this
    h = h % 1.0
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    
    i = i % 6
    if i == 0:   r, g, b = v, t, p
    elif i == 1: r, g, b = q, v, p
    elif i == 2: r, g, b = p, v, t
    elif i == 3: r, g, b = p, q, v
    elif i == 4: r, g, b = t, p, v
    else:        r, g, b = v, p, q
    
    return r, g, b

@jit(nopython=True, fastmath=True, parallel=True)
def apply_calm_tone_mapping(density_grid, surface_array, blend_factor, gamma_val, time_phase):
    for x in prange(PLOT_W):
        col_max = 0.0
        for y in range(PLOT_H):
            if density_grid[x, y] > col_max:
                col_max = density_grid[x, y]
        
        if col_max <= 0.0:
            col_max = 1.0

        inv_col_max = 1.0 / col_max
        x_norm = x / float(PLOT_W)

        for y in range(PLOT_H):
            val = density_grid[x, y]
            if val > 0.0:
                norm_val = val * inv_col_max
                density_factor = math.pow(norm_val, gamma_val)
                if density_factor > 1.0: density_factor = 1.0
                
                # charcoal esque to black
                base_r = 255.0 * (1.0 - density_factor)
                base_g = 255.0 * (1.0 - density_factor)
                base_b = 255.0 * (1.0 - density_factor)
                
                if blend_factor > 0.0:
                    y_norm = 1.0 - (y / float(PLOT_H))
                    
                    hue = (x_norm * 0.45 + y_norm * 0.2 + time_phase) % 1.0
                    sat = 0.45 
                    val_bright = 1.0 - (density_factor * 0.75) 
                    
                    chroma_r, chroma_g, chroma_b = hsv_to_rgb(hue, sat, val_bright)
                    chroma_r *= 255.0
                    chroma_g *= 255.0
                    chroma_b *= 255.0
                    
                    # cross fade effect
                    final_r = base_r + (chroma_r - base_r) * blend_factor
                    final_g = base_g + (chroma_g - base_g) * blend_factor
                    final_b = base_b + (chroma_b - base_b) * blend_factor
                else:
                    final_r, final_g, final_b = base_r, base_g, base_b
                
                surface_array[x, y, 0] = int(final_r)
                surface_array[x, y, 1] = int(final_g)
                surface_array[x, y, 2] = int(final_b)
            else:
                # pure white
                surface_array[x, y, 0] = 255
                surface_array[x, y, 1] = 255
                surface_array[x, y, 2] = 255

def format_val(val):
    if abs(val) < 1e-4 or abs(val) >= 1e4:
        return f"{val:.3e}"
    return f"{val:.4f}"

def draw_graph_axes(screen, rmin, rmax, xmin, xmax):
    bg_color = (255, 255, 255)
    axis_line_color = (0, 0, 0)
    tick_color = (60, 60, 60)
    grid_line_color = (235, 235, 235)
    text_color = (10, 10, 10)
    
    # filling margins with white
    pygame.draw.rect(screen, bg_color, (0, 0, MARGIN_LEFT, HEIGHT))
    pygame.draw.rect(screen, bg_color, (0, HEIGHT - MARGIN_BOTTOM, WIDTH, MARGIN_BOTTOM))
    pygame.draw.rect(screen, bg_color, (0, 0, WIDTH, MARGIN_TOP))
    pygame.draw.rect(screen, bg_color, (WIDTH - MARGIN_RIGHT, 0, MARGIN_RIGHT, HEIGHT))

    # border drawing
    pygame.draw.rect(screen, axis_line_color, (MARGIN_LEFT, MARGIN_TOP, PLOT_W, PLOT_H), width=2)

    # BOTTOM AXIS! (r parameter) ---
    num_ticks_r = 8
    for i in range(num_ticks_r + 1):
        frac = i / num_ticks_r
        px = MARGIN_LEFT + int(frac * PLOT_W)
        val = rmin + frac * (rmax - rmin)
        
        pygame.draw.line(screen, tick_color, (px, MARGIN_TOP + PLOT_H), (px, MARGIN_TOP + PLOT_H + 5), 1)
        if 0 < i < num_ticks_r:
            pygame.draw.line(screen, grid_line_color, (px, MARGIN_TOP), (px, MARGIN_TOP + PLOT_H), 1)
            
        lbl = font_small.render(format_val(val), True, text_color)
        screen.blit(lbl, (px - lbl.get_width() // 2, MARGIN_TOP + PLOT_H + 8))

    r_axis_lbl = font_axis.render("Parameter (r)", True, text_color)
    screen.blit(r_axis_lbl, (MARGIN_LEFT + PLOT_W // 2 - r_axis_lbl.get_width() // 2, HEIGHT - 18))

    # bottom r bar ---
    bar_x = MARGIN_LEFT
    bar_y = HEIGHT - 32
    bar_w = PLOT_W
    bar_h = 10
    pygame.draw.rect(screen, (220, 220, 220), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
    
    r_min_pct = (rmin - UNIVERSAL_R_MIN) / (UNIVERSAL_R_MAX - UNIVERSAL_R_MIN)
    r_max_pct = (rmax - UNIVERSAL_R_MIN) / (UNIVERSAL_R_MAX - UNIVERSAL_R_MIN)
    
    act_x1 = bar_x + int(r_min_pct * bar_w)
    act_x2 = bar_x + int(r_max_pct * bar_w)
    act_w = max(4, act_x2 - act_x1)
    
    pygame.draw.rect(screen, (50, 50, 50), (act_x1, bar_y, act_w, bar_h), border_radius=4)
    pygame.draw.circle(screen, (0, 0, 0), (act_x1, bar_y + bar_h // 2), 6)
    pygame.draw.circle(screen, (0, 0, 0), (act_x2, bar_y + bar_h // 2), 6)

    # population density!! ---
    num_ticks_x = 6
    for i in range(num_ticks_x + 1):
        frac = i / num_ticks_x
        py = MARGIN_TOP + PLOT_H - int(frac * PLOT_H)
        val = xmin + frac * (xmax - xmin)
        
        pygame.draw.line(screen, tick_color, (MARGIN_LEFT - 5, py), (MARGIN_LEFT, py), 1)
        if 0 < i < num_ticks_x:
            pygame.draw.line(screen, grid_line_color, (MARGIN_LEFT, py), (MARGIN_LEFT + PLOT_W, py), 1)

        lbl = font_small.render(format_val(val), True, text_color)
        screen.blit(lbl, (MARGIN_LEFT - lbl.get_width() - 18, py - lbl.get_height() // 2))

    x_axis_lbl = font_axis.render("Population (x)", True, text_color)
    x_axis_lbl = pygame.transform.rotate(x_axis_lbl, 90)
    screen.blit(x_axis_lbl, (10, MARGIN_TOP + PLOT_H // 2 - x_axis_lbl.get_height() // 2))

    # left density bar to show how zoomed in ---
    vbar_x = MARGIN_LEFT - 12
    vbar_y = MARGIN_TOP
    vbar_w = 10
    vbar_h = PLOT_H
    pygame.draw.rect(screen, (220, 220, 220), (vbar_x, vbar_y, vbar_w, vbar_h), border_radius=4)
    
    x_min_pct = (xmin - UNIVERSAL_X_MIN) / (UNIVERSAL_X_MAX - UNIVERSAL_X_MIN)
    x_max_pct = (xmax - UNIVERSAL_X_MIN) / (UNIVERSAL_X_MAX - UNIVERSAL_X_MIN)
    
    act_y1 = vbar_y + vbar_h - int(x_max_pct * vbar_h)
    act_y2 = vbar_y + vbar_h - int(x_min_pct * vbar_h)
    act_h = max(4, act_y2 - act_y1)
    
    pygame.draw.rect(screen, (50, 50, 50), (vbar_x, act_y1, vbar_w, act_h), border_radius=4)
    pygame.draw.circle(screen, (0, 0, 0), (vbar_x + vbar_w // 2, act_y1), 6)
    pygame.draw.circle(screen, (0, 0, 0), (vbar_x + vbar_w // 2, act_y2), 6)

def trigger_zoom(center_x, center_y, scale_factor):
    global target_rmin, target_rmax, target_xmin, target_xmax, is_animating
    
    norm_x = (center_x - MARGIN_LEFT) / PLOT_W
    norm_y = 1.0 - ((center_y - MARGIN_TOP) / PLOT_H)
    
    click_r = target_rmin + norm_x * (target_rmax - target_rmin)
    click_x = target_xmin + norm_y * (target_xmax - target_xmin)
    
    r_span = (target_rmax - target_rmin) * scale_factor
    x_span = (target_xmax - target_xmin) * scale_factor
    
    target_rmin = max(UNIVERSAL_R_MIN, click_r - r_span * norm_x)
    target_rmax = min(UNIVERSAL_R_MAX, click_r + r_span * (1.0 - norm_x))
    target_xmin = max(UNIVERSAL_X_MIN, click_x - x_span * norm_y)
    target_xmax = min(UNIVERSAL_X_MAX, click_x + x_span * (1.0 - norm_y))
    
    is_animating = True

def trigger_preset(pr_min, pr_max, px_min, px_max):
    global target_rmin, target_rmax, target_xmin, target_xmax, is_animating
    target_rmin, target_rmax = pr_min, pr_max
    target_xmin, target_xmax = px_min, px_max
    is_animating = True

# buffering
density_grid = np.zeros((PLOT_W, PLOT_H), dtype=np.float64)
surface_array = np.zeros((PLOT_W, PLOT_H, 3), dtype=np.uint8)
plot_surface = pygame.Surface((PLOT_W, PLOT_H))

running = True
is_animating = True
needs_redraw = True

# warmup JIT compiler
render_bifurcation_fast(density_grid, rmin, rmax, xmin, xmax, BASE_STEPS, BASE_SETTLE, BASE_PLOT)
apply_calm_tone_mapping(density_grid, surface_array, chroma_blend, gamma_curve, chroma_time)

while running:
    mx, my = pygame.mouse.get_pos()

    # interpolation (lerp is a good word)
    if chroma_active and chroma_blend < 1.0:
        chroma_blend = min(1.0, chroma_blend + BLEND_SPEED)
        needs_redraw = True
    elif not chroma_active and chroma_blend > 0.0:
        chroma_blend = max(0.0, chroma_blend - BLEND_SPEED)
        needs_redraw = True

    # gentle blend
    if chroma_blend > 0.0:
        chroma_time += 0.003
        needs_redraw = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and show_ui:
                bar_x, bar_y, bar_w, bar_h = MARGIN_LEFT, HEIGHT - 32, PLOT_W, 10
                if bar_y - 8 <= my <= bar_y + bar_h + 8 and bar_x <= mx <= bar_x + bar_w:
                    r_min_pct = (target_rmin - UNIVERSAL_R_MIN) / (UNIVERSAL_R_MAX - UNIVERSAL_R_MIN)
                    r_max_pct = (target_rmax - UNIVERSAL_R_MIN) / (UNIVERSAL_R_MAX - UNIVERSAL_R_MIN)
                    act_x1 = bar_x + int(r_min_pct * bar_w)
                    act_x2 = bar_x + int(r_max_pct * bar_w)
                    
                    if abs(mx - act_x1) <= 10:
                        drag_mode = 'r_min'
                    elif abs(mx - act_x2) <= 10:
                        drag_mode = 'r_max'

                vbar_x, vbar_y, vbar_w, vbar_h = MARGIN_LEFT - 12, MARGIN_TOP, 10, PLOT_H
                if vbar_x - 8 <= mx <= vbar_x + vbar_w + 8 and vbar_y <= my <= vbar_y + vbar_h:
                    x_min_pct = (target_xmin - UNIVERSAL_X_MIN) / (UNIVERSAL_X_MAX - UNIVERSAL_X_MIN)
                    x_max_pct = (target_xmax - UNIVERSAL_X_MIN) / (UNIVERSAL_X_MAX - UNIVERSAL_X_MIN)
                    act_y1 = vbar_y + vbar_h - int(x_max_pct * vbar_h)
                    act_y2 = vbar_y + vbar_h - int(x_min_pct * vbar_h)
                    
                    if abs(my - act_y1) <= 10:
                        drag_mode = 'x_max'
                    elif abs(my - act_y2) <= 10:
                        drag_mode = 'x_min'

                if drag_mode is None and MARGIN_LEFT <= mx <= WIDTH - MARGIN_RIGHT and MARGIN_TOP <= my <= HEIGHT - MARGIN_BOTTOM:
                    trigger_zoom(mx, my, 0.6)

            elif event.button in (3, 5):
                trigger_zoom(mx, my, 1.6)
            elif event.button == 4:
                trigger_zoom(mx, my, 0.6)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drag_mode = None

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_k:  # Toggle Calm Chroma Mode
                chroma_active = not chroma_active
            elif event.key == pygame.K_h:
                show_ui = not show_ui
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                gamma_curve = max(0.15, gamma_curve - 0.05)
                needs_redraw = True
            elif event.key == pygame.K_MINUS:
                gamma_curve = min(0.8, gamma_curve + 0.05)
                needs_redraw = True
            elif event.key == pygame.K_1:
                trigger_preset(2.0, 4.0, 0.0, 1.0)
            elif event.key == pygame.K_2:
                trigger_preset(2.95, 3.58, 0.2, 0.9)
            elif event.key == pygame.K_3:
                trigger_preset(3.82, 3.86, 0.1, 0.9)
            elif event.key == pygame.K_r:
                trigger_preset(2.0, 4.0, 0.0, 1.0)

    # more lerping
    if is_animating:
        rmin += (target_rmin - rmin) * tween_speed
        rmax += (target_rmax - rmax) * tween_speed
        xmin += (target_xmin - xmin) * tween_speed
        xmax += (target_xmax - xmax) * tween_speed

        if (abs(target_rmin - rmin) < EPSILON and 
            abs(target_rmax - rmax) < EPSILON and 
            abs(target_xmin - xmin) < EPSILON and 
            abs(target_xmax - xmax) < EPSILON):
            
            rmin, rmax = target_rmin, target_rmax
            xmin, xmax = target_xmin, target_xmax
            is_animating = False

        needs_redraw = True

    if needs_redraw:
        zoom_depth = 2.0 / float(rmax - rmin)
        detail_scale = 1.0 + math.log10(max(1.0, zoom_depth))
        
        steps = int(BASE_STEPS * detail_scale)
        settle_iters = int(BASE_SETTLE * detail_scale)
        plot_iters = int(BASE_PLOT * detail_scale)
        
        render_bifurcation_fast(density_grid, rmin, rmax, xmin, xmax, steps, settle_iters, plot_iters)
        apply_calm_tone_mapping(density_grid, surface_array, chroma_blend, gamma_curve, chroma_time)
        pygame.surfarray.blit_array(plot_surface, surface_array)
        needs_redraw = False

    # rendering
    if show_ui:
        screen.blit(plot_surface, (MARGIN_LEFT, MARGIN_TOP))
        draw_graph_axes(screen, rmin, rmax, xmin, xmax)
        
        hud_txt = font_ui.render("[K] Smooth Pastel Wave | [1,2,3] Presets | [R] Reset", True, (0, 0, 0))
        screen.blit(hud_txt, (MARGIN_LEFT, 3))
    else:
        scaled_plot = pygame.transform.scale(plot_surface, (WIDTH, HEIGHT))
        screen.blit(scaled_plot, (0, 0))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()