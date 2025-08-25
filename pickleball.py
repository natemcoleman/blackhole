import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle, Ellipse

# --- Physical constants ---
rho = 1.225
mass = 0.022
diameter = 0.074
R = diameter / 2
area = np.pi * R**2
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
# reach_radius_x = 1.5
# reach_radius_y = 2.5
arm_length = 0.5
reach_radius_x = arm_length
reach_radius_y = arm_length
shoulder_height = 1.4

# --- Aerodynamics functions ---
def getCdCl(speed, spin):
    omega = 2 * np.pi * spin
    r = diameter / 2
    S = omega * r / max(speed, 0.01)
    Cd0 = 0.45 if ball_type == 'indoor' else 0.33
    Cd = Cd0 + 0.02 * S * np.sign(spin)
    Cl0 = -0.05
    if spin > 0:
        Cl = Cl0 - 0.3 * S
    else:
        Cl = Cl0 + 0.05 * abs(S)
    return Cd, Cl

def dragAccel(vx, vy, cd):
    k = 0.5 * rho * cd * area / mass
    speed = max(1e-9, np.hypot(vx, vy))
    ax = -k * speed * vx
    ay = -k * speed * vy
    return ax, ay

def magnusAccel(vx, vy, cl):
    speed = max(1e-9, np.hypot(vx, vy))
    fl_over_m = 0.5 * rho * cl * area * speed**2 / mass
    ax = fl_over_m * (-vy / speed)
    ay = fl_over_m * (vx / speed)
    return ax, ay

def derivatives(state, spin):
    x, y, vx, vy = state
    speed = np.sqrt(vx**2 + vy**2)
    cd, cl = getCdCl(speed, spin)
    axD, ayD = dragAccel(vx, vy, cd)
    axL, ayL = magnusAccel(vx, vy, cl)
    ax = axD + axL
    ay = ayD - g + ayL
    return np.array([vx, vy, ax, ay])

def rk4_step(state, dt, spin):
    k1 = derivatives(state, spin)
    k2 = derivatives(state + 0.5*dt*k1, spin)
    k3 = derivatives(state + 0.5*dt*k2, spin)
    k4 = derivatives(state + dt*k3, spin)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


# --- Trajectory simulation ---
def simulateTrajectory(v0, angle, spin, player_x):
    x, y = 0, 1
    vx = v0 * np.cos(np.radians(angle))
    vy = v0 * np.sin(np.radians(angle))
    omega = 2*np.pi*spin
    state = np.array([x, y, vx, vy])
    sim_traj = [(x, y)]
    bounce_count = 0
    hit_status = "valid"

    while bounce_count < 2:
        prev_state = state.copy()
        state = rk4_step(state, 0.01, spin)
        x, y, vx, vy = state

        # --- Net check ---
        x_prev, y_prev = prev_state[0], prev_state[1]
        if (x_prev < net_x <= x) or (x_prev > net_x >= x):
            y_net = y_prev + (y - y_prev) * (net_x - x_prev) / (x - x_prev)
            if y_net <= net_height:
                sim_traj.append((net_x, y_net))
                return np.array(sim_traj), "net"

        # --- Bounce check ---
        if y < 0:
            bounce_count += 1
            if bounce_count == 1:
                # Check if bounce in bounds
                if not (0 <= x <= court_length):
                    return np.array(sim_traj), "out"
                # Apply bounce physics
                v_t = vx - omega * R
                delta_vx = -friction_factor * v_t
                vx += delta_vx
                omega += -delta_vx / R
                vy = -vy * restitution
                y = 0
                state = np.array([x, y, vx, vy])
            else:
                    break

        # --- Receiver check ---
        if (player_x - body_width / 2 <= x <= player_x + body_width / 2 and
                0 <= y <= body_height):
            return np.array(sim_traj), "body"
        dx = (x - player_x) / reach_radius_x
        # dy = (y - (body_height / 2)) / reach_radius_y
        dy = (y - shoulder_height) / reach_radius_y
        if dx ** 2 + dy ** 2 <= 1:
            return np.array(sim_traj), "reach"

        sim_traj.append((x, y))

    return np.array(sim_traj), hit_status


# --- Plot setup ---
fig, axes = plt.subplots(figsize=(10,5))
plt.subplots_adjust(bottom=0.35)

# Background court
axes.set_facecolor('#658553')
axes.axvspan(0, court_length, color='blue', alpha=0.3)
axes.fill_betweenx([0, net_height], net_position - 0.01, net_position + 0.01, color='orange')

# Kitchen lines
kitchen_length = 2.1336
kitchen_start = 4.572
axes.axvline(kitchen_start, 0, 5, color='white', lw=1)
axes.axvline(kitchen_start + (2*kitchen_length), 0, 5, color='white', lw=1)

# Initial trajectory
init_v0, init_angle, init_spin, init_player_x = 15, 20, 0, 12.5
init_traj, init_hit = simulateTrajectory(init_v0, init_angle, init_spin, init_player_x)
line, = axes.plot(init_traj[:,0], init_traj[:,1], lw=2, color='white')

# Receiver representation
body_patch = Rectangle((init_player_x-body_width/2,0), body_width, body_height, color="green", alpha=0.8)
# reach_patch = Ellipse((init_player_x, body_height/2), 2*reach_radius_x, 2*reach_radius_y, color="green", alpha=0.2)
reach_patch = Ellipse((init_player_x, shoulder_height), 2*reach_radius_x, 2*reach_radius_y, color="green", alpha=0.2)
axes.add_patch(body_patch)
axes.add_patch(reach_patch)

# Axes settings
axes.set_xlabel("Distance (m)")
axes.set_ylabel("Height (m)")
axes.set_xlim(-1, court_length+1)
axes.set_ylim(0, 5)
axes.set_aspect('equal', adjustable='box')
axes.grid(True)

# --- Sliders ---
ax_v0 = plt.axes([0.2, 0.25, 0.65, 0.03])
ax_angle = plt.axes([0.2, 0.2, 0.65, 0.03])
ax_spin = plt.axes([0.2, 0.15, 0.65, 0.03])
ax_player = plt.axes([0.2, 0.1, 0.65, 0.03])

s_v0 = Slider(ax_v0, "Velocity", 5, 25, valinit=init_v0)
s_angle = Slider(ax_angle, "Angle", 0, 90, valinit=init_angle)
s_spin = Slider(ax_spin, "Spin", -100, 100, valinit=init_spin)
s_player = Slider(ax_player, "Player X", net_x+1, court_length, valinit=init_player_x)

def update(val):
    v0 = s_v0.val
    angle = s_angle.val
    spin = s_spin.val
    player_x = s_player.val
    traj, hit_status = simulateTrajectory(v0, angle, spin, player_x)
    line.set_xdata(traj[:,0])
    line.set_ydata(traj[:,1])

    # Update receiver position
    body_patch.set_xy((player_x-body_width/2, 0))
    # reach_patch.center = (player_x, body_height/2)
    reach_patch.center = (player_x, shoulder_height)

    # Reset colors
    body_patch.set_color("green")
    body_patch.set_alpha(0.8)
    reach_patch.set_color("green")
    reach_patch.set_alpha(0.2)

    # Change colors depending on result
    if hit_status == "valid":
        line.set_color("white")
    elif hit_status == "net":
        line.set_color("red")
    elif hit_status == "out":
        line.set_color("red")
    elif hit_status == "body":
        line.set_color("red")
        body_patch.set_color("red")
    elif hit_status == "reach":
        line.set_color("red")
        reach_patch.set_color("red")

    fig.canvas.draw_idle()


update(0)  # Initial update to set the plot

s_v0.on_changed(update)
s_angle.on_changed(update)
s_spin.on_changed(update)
s_player.on_changed(update)

plt.show()
