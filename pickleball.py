import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- Physical constants ---
rho = 1.225
mass = 0.022
diameter = 0.074
area = np.pi * (diameter / 2) ** 2
g = 9.81
ball_type = 'indoor'

# --- Court dimensions ---
court_length = 13.41
court_width = 6.10
net_position = court_length / 2
net_height = 0.91


# --- Aerodynamics functions ---
def getCdCl(speed, spin):
    omega = 2 * np.pi * spin
    r = diameter / 2
    S = omega * r / max(speed, 0.01)
    Cd0 = 0.45 if ball_type == 'indoor' else 0.33
    Cd = Cd0 + 0.02 * S * np.sign(spin)
    # if spin > 0:
    #     Cl = 0.24 + 0.65 * S
    #     # Cl *= -1
    # else:
    #     Cl = 0.18 + 0.042 * S
    Cl0 = -0.05
    if spin > 0:
        Cl = Cl0 - 0.3 * S
    else:
        Cl = Cl0 + 0.05 * abs(S)
    return Cd, Cl


def dragAccel(vx, vy, cd, useComponentDrag=False):
    k = 0.5 * rho * cd * area / mass
    if useComponentDrag:
        ax = -k * abs(vx) * vx
        ay = -k * abs(vy) * vy
    else:
        speed = max(1e-9, np.hypot(vx, vy))
        ax = -k * speed * vx
        ay = -k * speed * vy
    return ax, ay


def magnusAccel(vx, vy, cl):
    speed = max(1e-9, np.hypot(vx, vy))
    fl_over_m = 0.5 * rho * cl * area * speed ** 2 / mass
    ax = fl_over_m * (-vy / speed)
    ay = fl_over_m * (vx / speed)
    return ax, ay


def simulateTrajectory(v0, launchAngleDeg, spin, useComponentDrag=False):
    angle = np.radians(launchAngleDeg)
    vx, vy = v0 * np.cos(angle), v0 * np.sin(angle)
    x, y = 0.0, 0.0
    dt = 0.01
    restitution = 0.65  # energy retained after bounce
    bounce_count = 0
    net_x = 6.705  # meters
    net_height = 0.86  # meters at center
    traj = [(x, y)]
    # while y >= 0:
    while bounce_count < 2:
        x_prev, y_prev = x, y
        speed = max(1e-9, np.hypot(vx, vy))
        if speed == 0:
            break
        cd, cl = getCdCl(speed, spin)
        axD, ayD = dragAccel(vx, vy, cd, useComponentDrag)
        axL, ayL = magnusAccel(vx, vy, cl)
        ax = axD + axL
        ay = ayD - g + ayL
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt

        # --- Net check ---
        if (x_prev < net_x <= x) or (x_prev > net_x >= x):
            y_at_net = y_prev + (y - y_prev) * (net_x - x_prev) / (x - x_prev)
            if y_at_net <= net_height:
                traj.append((net_x, y_at_net))
                break  # ball hit the net

        if y < 0:  # ground contact
            bounce_count += 1
            if bounce_count == 1:
                y = 0
                vy = -vy * restitution
                # vx *= 0.9
                vx *= 1
            else:
                break  # second bounce, stop

        traj.append((x, y))
    return np.array(traj)


# --- Plot setup ---
fig, ax = plt.subplots(figsize=(10, 5))
plt.subplots_adjust(bottom=0.3)

# green background
ax.set_facecolor('#658553')

# blue court area
ax.axvspan(0, court_length, color='blue', alpha=0.3)

# net representation
net_patch = ax.fill_betweenx([0, net_height], net_position - 0.01, net_position + 0.01, color='orange')

# draw kitchen area
kitchen_length = 2.1336  # meters
kitchen_start = 4.572
kitchen_height = 5
ax.axvline(kitchen_start, 0, kitchen_height, color='white', lw=1)  # approximate relative height
ax.axvline(kitchen_start + (2*kitchen_length), 0, kitchen_height, color='white', lw=1)  # approximate relative height


# initial trajectory
init_v0, init_angle, init_spin = 15, 20, 0
traj = simulateTrajectory(init_v0, init_angle, init_spin)
[line] = ax.plot(traj[:, 0], traj[:, 1], lw=2, color='white')

# axes labels and grid
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Height (m)")
ax.grid(True)
ax.set_aspect('equal', adjustable='box')
ax.set_ylim(bottom=0)
padding = 1
ax.set_xlim(-padding, court_length+padding)
ax.set_ylim(0, 5)
plt.grid(False)

# --- Sliders ---
ax_v0 = plt.axes([0.2, 0.2, 0.65, 0.03])
ax_angle = plt.axes([0.2, 0.15, 0.65, 0.03])
ax_spin = plt.axes([0.2, 0.1, 0.65, 0.03])
s_v0 = Slider(ax_v0, "Velocity", 5, 25, valinit=init_v0)
s_angle = Slider(ax_angle, "Angle", 0, 90, valinit=init_angle)
s_spin = Slider(ax_spin, "Spin", -25, 25, valinit=init_spin)


def update(val):
    v0 = s_v0.val
    angle = s_angle.val
    spin = s_spin.val
    traj = simulateTrajectory(v0, angle, spin)
    line.set_xdata(traj[:, 0])
    line.set_ydata(traj[:, 1])

    # update limits dynamically with padding
    padding = 1
    xmax = max(court_length, np.max(traj[:, 0]) * 1.05)
    ymax = max(5, np.max(traj[:, 1]) * 1.05)
    ax.set_xlim(-padding, xmax+padding)
    ax.set_ylim(0, ymax)
    ax.set_aspect('equal', adjustable='box')

    # redraw court and net
    # remove old court patches
    for patch in ax.patches[:]:
        patch.remove()
    for coll in ax.collections[:]:
        coll.remove()
    ax.axvspan(0, court_length, color='blue', alpha=0.3)
    net_patch = ax.fill_betweenx([0, net_height], net_position - 0.01, net_position + 0.01, color='orange')

    fig.canvas.draw_idle()


s_v0.on_changed(update)
s_angle.on_changed(update)
s_spin.on_changed(update)

plt.show()
