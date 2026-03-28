from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk

DATA_DIR = Path("data")


def download_datasets(max_math_stories: int = 500_000, max_stories: int = 500_000, max_analogies: int = 500_000):
    """Download datasets, streaming Cosmopedia to only fetch needed rows. Caches to data/."""
    DATA_DIR.mkdir(exist_ok=True)

    math_stories = load_dataset("azminetoushikwasi/math-story-problems")
    analogies = load_dataset("saturnMars/hyperprobe-dataset-analogy")

    # Cap rows per split
    for split in math_stories:
        n = len(math_stories[split])
        if n > max_math_stories:
            math_stories[split] = math_stories[split].select(range(max_math_stories))
    for split in analogies:
        n = len(analogies[split])
        if n > max_analogies:
            analogies[split] = analogies[split].select(range(max_analogies))

    # Cosmopedia: stream only the rows we need, cache to disk
    stories_path = DATA_DIR / "cosmopedia_stories"
    if stories_path.exists():
        stories_ds = load_from_disk(str(stories_path))
    else:
        stream = load_dataset("HuggingFaceTB/cosmopedia", "stories", split="train", streaming=True)
        stories_ds = Dataset.from_list(list(stream.take(max_stories)))
        stories_ds.save_to_disk(str(stories_path))

    return {
        "math_stories": math_stories,
        "stories": {"train": stories_ds},
        "analogies": analogies,
    }


if __name__ == "__main__":
    download_datasets()