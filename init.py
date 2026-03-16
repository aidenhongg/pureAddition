from datasets import load_dataset


def download_datasets(max_math_stories: int = 500_000, max_tiny_stories: int = 500_000):
    math_stories = load_dataset("azminetoushikwasi/math-story-problems")
    tiny_stories = load_dataset("roneneldan/TinyStories")

    # Cap rows per split
    for split in math_stories:
        n = len(math_stories[split])
        if n > max_math_stories:
            math_stories[split] = math_stories[split].select(range(max_math_stories))
    for split in tiny_stories:
        n = len(tiny_stories[split])
        if n > max_tiny_stories:
            tiny_stories[split] = tiny_stories[split].select(range(max_tiny_stories))

    return {"math_stories": math_stories, "tiny_stories": tiny_stories}


if __name__ == "__main__":
    download_datasets()