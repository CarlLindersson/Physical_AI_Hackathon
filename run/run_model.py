#-----------------------------------------------------------------------------#
#--------------------- Keyboard Control - QArm Mini ---------------------------#
#-----------------------------------------------------------------------------#

import numpy as np
import pygame
from pal.products.qarm_mini import QArmMini
from pal.utilities.timing import QTimer


SAMPLE_RATE_HZ = 30.0
RUN_TIME_SECONDS = 300.0
JOINT_SPEED_RAD_PER_SEC = np.pi / 4

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.0


def print_controls():
    print("\nQArm Mini keyboard control")
    print("=" * 32)
    print("  Up/Down     shoulder up/down")
    print("  Left/Right  base left/right")
    print("  p           close gripper")
    print("  o           open gripper")
    print("  h           home position")
    print("  q or Esc    quit")
    print("\nClick/focus the pygame keyboard window before driving the arm.\n")


def draw_keyboard_window(font, gripper_cmd):
    screen = pygame.display.get_surface()
    screen.fill((20, 22, 26))

    lines = [
        "QArm Mini Keyboard Control",
        "Arrows: base left/right, shoulder up/down",
        "p: close gripper    o: open gripper",
        "h: home    q/Esc: quit",
        f"Gripper: {'closed' if gripper_cmd == GRIPPER_CLOSED else 'open'}",
    ]

    for index, line in enumerate(lines):
        color = (240, 240, 240) if index == 0 else (190, 196, 205)
        text = font.render(line, True, color)
        screen.blit(text, (18, 18 + index * 24))

    pygame.display.flip()


def handle_keydown_events(joint_cmd, gripper_cmd):
    should_quit = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            should_quit = True
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                should_quit = True
            elif event.key == pygame.K_p:
                gripper_cmd = GRIPPER_CLOSED
                print("Gripper: closed")
            elif event.key == pygame.K_o:
                gripper_cmd = GRIPPER_OPEN
                print("Gripper: open")
            elif event.key == pygame.K_h:
                joint_cmd[:] = QArmMini.HOME_POSE
                print("Moving to home position")

    return should_quit, gripper_cmd


def apply_arrow_key_motion(joint_cmd, timestep):
    keys = pygame.key.get_pressed()
    step = JOINT_SPEED_RAD_PER_SEC * timestep

    joint_cmd[0] += (int(keys[pygame.K_LEFT]) - int(keys[pygame.K_RIGHT])) * step
    joint_cmd[1] += (int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN])) * step

    np.clip(joint_cmd, QArmMini.LIMITS_MIN, QArmMini.LIMITS_MAX, out=joint_cmd)


def main():
    print_controls()

    pygame.init()
    pygame.display.set_mode((460, 150))
    pygame.display.set_caption("QArm Mini Keyboard Control")
    font = pygame.font.Font(None, 24)

    myMiniArm = QArmMini(hardware=1, id=3)
    timer = QTimer(sampleRate=SAMPLE_RATE_HZ, totalTime=RUN_TIME_SECONDS)

    joint_cmd = QArmMini.HOME_POSE.copy()
    gripper_cmd = GRIPPER_OPEN

    try:
        while timer.check():
            should_quit, gripper_cmd = handle_keydown_events(joint_cmd, gripper_cmd)
            if should_quit:
                break

            apply_arrow_key_motion(joint_cmd, timer.get_sample_time())
            myMiniArm.read_write_std(joint_cmd, gripper_cmd)
            draw_keyboard_window(font, gripper_cmd)

            timer.sleep()

    except KeyboardInterrupt:
        print("Received user terminate command.")

    finally:
        myMiniArm.terminate()
        pygame.quit()


if __name__ == "__main__":
    main()
