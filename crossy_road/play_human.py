from __future__ import annotations

from crossy_road import CrossyRoadEnv


def main() -> None:
    env = CrossyRoadEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    episode = 1

    try:
        while True:
            action = env.get_human_action()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(
                    {
                        "episode": episode,
                        "finished": info.get("finished", False),
                        "collision": info.get("collision", False),
                        "steps": info.get("steps"),
                        "progress": round(info.get("progress", 0.0), 3),
                    }
                )
                episode += 1
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
