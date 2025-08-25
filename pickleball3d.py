import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle, Ellipse, Polygon
from mpl_toolkits.mplot3d import Axes3D

# --- Physical constants ---
rho = 1.225
mass = 0.022
diameter = 0.074
R = diameter / 2
area = np.pi * R ** 2
g = 9.81
ball_type = 'indoor'

# --- Court dimensions ---
court_length = 13.41
court_width = 6.10
net_position = court_length / 2
net_height = 0.91
net_x = net_position

# Bounce params
restitution = 0.45
friction_factor = 0.25

# Receiver dimensions
body_width = 0.4
body_height = 1.8
arm_length = 0.5
reach_radius_x = arm_length
reach_radius_y = arm_length
reach_radius_z = arm_length
shoulder_height = 1.4


# --- Aerodynamics functions ---
def getCdCl(speed, spin_total):
    omega = 2 * np.pi * spin_total
    r = diameter / 2
    S = omega * r / max(speed, 0.01)
    Cd0 = 0.45 if ball_type == 'indoor' else 0.33
    Cd = Cd0 + 0.02 * S * np.sign(spin_total)
    Cl0 = -0.05
    if spin_total > 0:
        Cl = Cl0 - 0.3 * S
    else:
        Cl = Cl0 + 0.05 * abs(S)
    return Cd, Cl


def dragAccel(vx, vy, vz, cd):
    k = 0.5 * rho * cd * area / mass
    speed = max(1e-9, np.sqrt(vx ** 2 + vy ** 2 + vz ** 2))
    ax = -k * speed * vx
    ay = -k * speed * vy
    az = -k * speed * vz
    return ax, ay, az


def magnusAccel(vx, vy, vz, spin_x, spin_y, spin_z):
    # Magnus force: F = (1/2) * rho * v^2 * A * Cl * (spin x v) / |spin x v|
    # For simplicity, we'll treat each spin component separately
    speed = max(1e-9, np.sqrt(vx ** 2 + vy ** 2 + vz ** 2))

    # Total spin magnitude for Cl calculation
    spin_total = np.sqrt(spin_x ** 2 + spin_y ** 2 + spin_z ** 2)
    _, cl = getCdCl(speed, spin_total)

    # Magnus acceleration components
    omega = 2 * np.pi
    ax_mag = omega * (spin_y * vz - spin_z * vy) * cl * 0.5 * rho * area * speed / mass
    ay_mag = omega * (spin_z * vx - spin_x * vz) * cl * 0.5 * rho * area * speed / mass
    az_mag = omega * (spin_x * vy - spin_y * vx) * cl * 0.5 * rho * area * speed / mass

    return ax_mag, ay_mag, az_mag


def derivatives(state, spin_x, spin_y, spin_z):
    x, y, z, vx, vy, vz = state
    speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    spin_total = np.sqrt(spin_x ** 2 + spin_y ** 2 + spin_z ** 2)
    cd, _ = getCdCl(speed, spin_total)

    axD, ayD, azD = dragAccel(vx, vy, vz, cd)
    axM, ayM, azM = magnusAccel(vx, vy, vz, spin_x, spin_y, spin_z)

    ax = axD + axM
    ay = ayD - g + ayM
    az = azD + azM

    return np.array([vx, vy, vz, ax, ay, az])


def rk4_step(state, dt, spin_x, spin_y, spin_z):
    k1 = derivatives(state, spin_x, spin_y, spin_z)
    k2 = derivatives(state + 0.5 * dt * k1, spin_x, spin_y, spin_z)
    k3 = derivatives(state + 0.5 * dt * k2, spin_x, spin_y, spin_z)
    k4 = derivatives(state + dt * k3, spin_x, spin_y, spin_z)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# --- 3D Trajectory simulation ---
def simulateTrajectory3D(v0, angle_elevation, angle_azimuth, spin_x, spin_y, spin_z, player_x, player_z):
    # Starting position (serving from baseline)
    x, y, z = 0, 1, court_width / 2

    # Initial velocity components
    vx = v0 * np.cos(np.radians(angle_elevation)) * np.cos(np.radians(angle_azimuth))
    vy = v0 * np.sin(np.radians(angle_elevation))
    vz = v0 * np.cos(np.radians(angle_elevation)) * np.sin(np.radians(angle_azimuth))

    state = np.array([x, y, z, vx, vy, vz])
    sim_traj = [(x, y, z)]
    bounce_count = 0
    hit_status = "valid"
    dt = 0.01

    while bounce_count < 2 and len(sim_traj) < 5000:  # Safety limit
        prev_state = state.copy()
        state = rk4_step(state, dt, spin_x, spin_y, spin_z)
        x, y, z, vx, vy, vz = state

        # --- Net check ---
        x_prev, y_prev = prev_state[0], prev_state[1]
        if (x_prev < net_x <= x) or (x_prev > net_x >= x):
            y_net = y_prev + (y - y_prev) * (net_x - x_prev) / (x - x_prev)
            if y_net <= net_height:
                z_net = prev_state[2] + (z - prev_state[2]) * (net_x - x_prev) / (x - x_prev)
                sim_traj.append((net_x, y_net, z_net))
                return np.array(sim_traj), "net"

        # --- Bounce check ---
        if y < 0:
            bounce_count += 1
            if bounce_count == 1:
                # Check if bounce in bounds
                if not (0 <= x <= court_length and 0 <= z <= court_width):
                    return np.array(sim_traj), "out"

                # Apply bounce physics (simplified for 3D)
                vy = -vy * restitution
                vx *= (1 - friction_factor * 0.1)  # Simplified friction
                vz *= (1 - friction_factor * 0.1)
                y = 0
                state = np.array([x, y, z, vx, vy, vz])
            else:
                break

        # --- Receiver check ---
        if (player_x - body_width / 2 <= x <= player_x + body_width / 2 and
                player_z - body_width / 2 <= z <= player_z + body_width / 2 and
                0 <= y <= body_height):
            return np.array(sim_traj), "body"

        # Check if within reach
        dx = (x - player_x) / reach_radius_x
        dy = (y - shoulder_height) / reach_radius_y
        dz = (z - player_z) / reach_radius_z
        if dx ** 2 + dy ** 2 + dz ** 2 <= 1:
            return np.array(sim_traj), "reach"

        sim_traj.append((x, y, z))

    return np.array(sim_traj), hit_status


# --- Court drawing functions ---
def draw_court_lines_3d(ax):
    # Court surface (ground plane)
    court_x = np.array([[0, court_length], [0, court_length]])
    court_y = np.array([[0, 0], [court_width, court_width]])
    court_z = np.array([[0, 0], [0, 0]])
    ax.plot_surface(court_x, court_y, court_z, alpha=0.3, color='#658553')

    # Court boundary lines
    court_corners = np.array([
        [0, 0, 0], [court_length, 0, 0],
        [court_length, court_width, 0], [0, court_width, 0], [0, 0, 0]
    ])
    ax.plot(court_corners[:, 0], court_corners[:, 1], court_corners[:, 2], 'w-', lw=3)

    # Net
    ax.plot([net_x, net_x], [0, court_width], [0, 0], 'orange', lw=4)
    ax.plot([net_x, net_x], [0, court_width], [net_height, net_height], 'orange', lw=4)
    ax.plot([net_x, net_x], [0, 0], [0, net_height], 'orange', lw=4)
    ax.plot([net_x, net_x], [court_width, court_width], [0, net_height], 'orange', lw=4)

    # Kitchen lines
    kitchen_length = 2.1336
    kitchen_start = net_x - kitchen_length
    kitchen_end = net_x + kitchen_length
    ax.plot([kitchen_start, kitchen_start], [0, court_width], [0, 0], 'w-', lw=2)
    ax.plot([kitchen_end, kitchen_end], [0, court_width], [0, 0], 'w-', lw=2)

    # Center line in kitchen
    ax.plot([kitchen_start, kitchen_end], [court_width / 2, court_width / 2], [0, 0], 'w-', lw=2)


def draw_court_lines_side(ax):
    # Court ground
    ax.axhline(0, color='white', lw=2)
    ax.axvline(0, color='white', lw=2)
    ax.axvline(court_length, color='white', lw=2)

    # Net
    ax.plot([net_x, net_x], [0, net_height], 'orange', lw=3)

    # Kitchen lines
    kitchen_length = 2.1336
    kitchen_start = net_x - kitchen_length
    kitchen_end = net_x + kitchen_length
    ax.axvline(kitchen_start, color='white', lw=1)
    ax.axvline(kitchen_end, color='white', lw=1)


def draw_court_lines_top(ax):
    # Court boundary
    court_rect = Rectangle((0, 0), court_length, court_width,
                           linewidth=2, edgecolor='white', facecolor='none')
    ax.add_patch(court_rect)

    # Net line
    ax.plot([net_x, net_x], [0, court_width], 'orange', lw=3)

    # Kitchen lines
    kitchen_length = 2.1336
    kitchen_start = net_x - kitchen_length
    kitchen_end = net_x + kitchen_length
    ax.axvline(kitchen_start, color='white', lw=1)
    ax.axvline(kitchen_end, color='white', lw=1)

    # Center line in kitchen
    ax.plot([kitchen_start, kitchen_end], [court_width / 2, court_width / 2], 'white', lw=1)


# --- Plot setup ---
fig = plt.figure(figsize=(18, 10))
plt.subplots_adjust(bottom=0.25, hspace=0.3, wspace=0.3)

# Create subplots - 3D on left, 2D plots stacked on right
ax3d = fig.add_subplot(1, 2, 1, projection='3d')
ax_side = fig.add_subplot(2, 2, 2)
ax_top = fig.add_subplot(2, 2, 4)

# Position plots manually for better layout
ax3d.set_position([0.05, 0.3, 0.45, 0.65])  # [left, bottom, width, height]
ax_side.set_position([0.55, 0.55, 0.4, 0.35])
ax_top.set_position([0.55, 0.3, 0.4, 0.2])

# Set backgrounds
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor('white')
ax3d.yaxis.pane.set_edgecolor('white')
ax3d.zaxis.pane.set_edgecolor('white')
ax3d.xaxis.pane.set_alpha(0.1)
ax3d.yaxis.pane.set_alpha(0.1)
ax3d.zaxis.pane.set_alpha(0.1)
ax_side.set_facecolor('#658553')
ax_top.set_facecolor('#658553')

# Initial parameters
init_v0 = 15
init_elevation = 20
init_azimuth = 0
init_spin_x = 0
init_spin_y = 0
init_spin_z = 0
init_player_x = 12.5
init_player_z = 3.0

# Initial trajectory
init_traj, init_hit = simulateTrajectory3D(init_v0, init_elevation, init_azimuth,
                                           init_spin_x, init_spin_y, init_spin_z,
                                           init_player_x, init_player_z)

# Plot initial trajectory
line3d, = ax3d.plot(init_traj[:, 0], init_traj[:, 2], init_traj[:, 1], 'w-', lw=3)
line_side, = ax_side.plot(init_traj[:, 0], init_traj[:, 1], 'w-', lw=2)
line_top, = ax_top.plot(init_traj[:, 0], init_traj[:, 2], 'w-', lw=2)

# Draw courts
draw_court_lines_3d(ax3d)
draw_court_lines_side(ax_side)
draw_court_lines_top(ax_top)


# Add 3D player representation
def draw_player_3d(ax, x, z, color="green", alpha=0.8):
    # Player body as a vertical cylinder
    theta = np.linspace(0, 2 * np.pi, 20)
    body_radius = body_width / 2

    # Body cylinder
    x_body = x + body_radius * np.cos(theta)
    z_body = z + body_radius * np.sin(theta)
    y_bottom = np.zeros_like(x_body)
    y_top = np.full_like(x_body, body_height)

    # Draw body cylinder walls
    for i in range(len(theta) - 1):
        ax.plot([x_body[i], x_body[i + 1]], [z_body[i], z_body[i + 1]], [0, 0], color=color, alpha=alpha, lw=2)
        ax.plot([x_body[i], x_body[i + 1]], [z_body[i], z_body[i + 1]], [body_height, body_height], color=color,
                alpha=alpha, lw=2)
        ax.plot([x_body[i], x_body[i]], [z_body[i], z_body[i]], [0, body_height], color=color, alpha=alpha, lw=1)

    # Reach sphere (wireframe)
    u = np.linspace(0, np.pi, 10)
    v = np.linspace(0, 2 * np.pi, 20)
    x_sphere = x + reach_radius_x * np.outer(np.sin(u), np.cos(v))
    y_sphere = shoulder_height + reach_radius_y * np.outer(np.cos(u), np.ones(np.size(v)))
    z_sphere = z + reach_radius_z * np.outer(np.sin(u), np.sin(v))

    # Draw wireframe sphere for reach
    ax.plot_wireframe(x_sphere, z_sphere, y_sphere, alpha=0.1, color=color, linewidth=0.5)


# Draw initial player
player_3d_elements = []

# Add receiver representations for 2D views
body_patch_top = Rectangle((init_player_x - body_width / 2, init_player_z - body_width / 2),
                           body_width, body_width, color="green", alpha=0.8)
reach_patch_top = Ellipse((init_player_x, init_player_z), 2 * reach_radius_x, 2 * reach_radius_z,
                          color="green", alpha=0.2)
ax_top.add_patch(body_patch_top)
ax_top.add_patch(reach_patch_top)

body_patch_side = Rectangle((init_player_x - body_width / 2, 0), body_width, body_height,
                            color="green", alpha=0.8)
reach_patch_side = Ellipse((init_player_x, shoulder_height), 2 * reach_radius_x, 2 * reach_radius_y,
                           color="green", alpha=0.2)
ax_side.add_patch(body_patch_side)
ax_side.add_patch(reach_patch_side)

# Set axis properties
ax3d.set_xlabel('Length (m)')
ax3d.set_ylabel('Width (m)')
ax3d.set_zlabel('Height (m)')
ax3d.set_xlim(0, court_length)
ax3d.set_ylim(0, court_width)
ax3d.set_zlim(0, 5)

# Set initial view angle and restrict elevation
ax3d.view_init(elev=35, azim=-135)


# Restrict the elevation angle to prevent going below ground
def on_move(event):
    if event.inaxes == ax3d:
        elev, azim = ax3d.elev, ax3d.azim
        if elev < 0:
            ax3d.view_init(elev=0, azim=azim)
            plt.draw()


fig.canvas.mpl_connect('motion_notify_event', on_move)

ax_side.set_xlabel('Length (m)')
ax_side.set_ylabel('Height (m)')
ax_side.set_xlim(-1, court_length + 1)
ax_side.set_ylim(0, 5)
ax_side.set_aspect('equal', adjustable='box')
ax_side.grid(True)

ax_top.set_xlabel('Length (m)')
ax_top.set_ylabel('Width (m)')
ax_top.set_xlim(-1, court_length + 1)
ax_top.set_ylim(-1, court_width + 1)
ax_top.set_aspect('equal', adjustable='box')
ax_top.grid(True)

# --- Sliders ---
slider_height = 0.025
slider_width = 0.35
slider_spacing = 0.035
start_y = 0.18

# Left column sliders - adjusted positioning to fit labels
ax_v0 = plt.axes([0.08, start_y, slider_width, slider_height])
ax_elev = plt.axes([0.08, start_y - slider_spacing, slider_width, slider_height])
ax_azim = plt.axes([0.08, start_y - 2 * slider_spacing, slider_width, slider_height])
ax_spin_x = plt.axes([0.08, start_y - 3 * slider_spacing, slider_width, slider_height])

# Right column sliders
ax_spin_y = plt.axes([0.55, start_y, slider_width, slider_height])
ax_spin_z = plt.axes([0.55, start_y - slider_spacing, slider_width, slider_height])
ax_player_x = plt.axes([0.55, start_y - 2 * slider_spacing, slider_width, slider_height])
ax_player_z = plt.axes([0.55, start_y - 3 * slider_spacing, slider_width, slider_height])

s_v0 = Slider(ax_v0, "Velocity (m/s)", 5, 30, valinit=init_v0)
s_elev = Slider(ax_elev, "Elevation (°)", -10, 60, valinit=init_elevation)
s_azim = Slider(ax_azim, "Azimuth (°)", -45, 45, valinit=init_azimuth)
s_spin_x = Slider(ax_spin_x, "Spin X (rps)", -50, 50, valinit=init_spin_x)
s_spin_y = Slider(ax_spin_y, "Spin Y (rps)", -50, 50, valinit=init_spin_y)
s_spin_z = Slider(ax_spin_z, "Spin Z (rps)", -50, 50, valinit=init_spin_z)
s_player_x = Slider(ax_player_x, "Player X (m)", net_x + 1, court_length, valinit=init_player_x)
s_player_z = Slider(ax_player_z, "Player Z (m)", 0.5, court_width - 0.5, valinit=init_player_z)


def update(val):
    v0 = s_v0.val
    elevation = s_elev.val
    azimuth = s_azim.val
    spin_x = s_spin_x.val
    spin_y = s_spin_y.val
    spin_z = s_spin_z.val
    player_x = s_player_x.val
    player_z = s_player_z.val

    traj, hit_status = simulateTrajectory3D(v0, elevation, azimuth, spin_x, spin_y, spin_z,
                                            player_x, player_z)

    # Clear 3D plot and redraw everything
    ax3d.clear()

    # Redraw court
    draw_court_lines_3d(ax3d)

    # Redraw trajectory
    ax3d.plot(traj[:, 0], traj[:, 2], traj[:, 1], lw=3,
              color="red" if hit_status in ["net", "out", "body", "reach"] else "black")

    # Redraw player in 3D
    player_color = "red" if hit_status in ["body", "reach"] else "green"
    draw_player_3d(ax3d, player_x, player_z, color=player_color)

    # Reset 3D axis properties
    ax3d.set_xlabel('Length (m)')
    ax3d.set_ylabel('Width (m)')
    ax3d.set_zlabel('Height (m)')
    ax3d.set_xlim(0, court_length)
    ax3d.set_ylim(0, court_width)
    ax3d.set_zlim(0, 5)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor('white')
    ax3d.yaxis.pane.set_edgecolor('white')
    ax3d.zaxis.pane.set_edgecolor('white')
    ax3d.xaxis.pane.set_alpha(0.1)
    ax3d.yaxis.pane.set_alpha(0.1)
    ax3d.zaxis.pane.set_alpha(0.1)

    # Update 2D trajectory lines
    line_side.set_xdata(traj[:, 0])
    line_side.set_ydata(traj[:, 1])
    line_top.set_xdata(traj[:, 0])
    line_top.set_ydata(traj[:, 2])

    # Update receiver positions in 2D views
    body_patch_top.set_xy((player_x - body_width / 2, player_z - body_width / 2))
    reach_patch_top.center = (player_x, player_z)
    body_patch_side.set_xy((player_x - body_width / 2, 0))
    reach_patch_side.center = (player_x, shoulder_height)

    # Set colors for 2D views
    color = "red" if hit_status in ["net", "out", "body", "reach"] else "white"
    body_color = "red" if hit_status in ["body", "reach"] else "green"
    reach_color = "red" if hit_status == "reach" else "green"

    line_side.set_color(color)
    line_top.set_color(color)

    body_patch_top.set_color(body_color)
    body_patch_top.set_alpha(0.8)
    reach_patch_top.set_color(reach_color)
    reach_patch_top.set_alpha(0.2)
    body_patch_side.set_color(body_color)
    body_patch_side.set_alpha(0.8)
    reach_patch_side.set_color(reach_color)
    reach_patch_side.set_alpha(0.2)

    fig.canvas.draw_idle()


# Connect sliders
s_v0.on_changed(update)
s_elev.on_changed(update)
s_azim.on_changed(update)
s_spin_x.on_changed(update)
s_spin_y.on_changed(update)
s_spin_z.on_changed(update)
s_player_x.on_changed(update)
s_player_z.on_changed(update)

# Initial update
update(0)
draw_player_3d(ax3d, init_player_x, init_player_z)

plt.suptitle('3D Pickleball Trajectory Simulator', fontsize=18, y=0.98)

plt.show()