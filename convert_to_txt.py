#!/usr/bin/env python3
"""
Convert training data from parquet format to text file.
Specifically designed to convert TinyStories dataset for language model training.
"""

import argparse
import pandas as pd
import sys
from pathlib import Path
import tqdm


def read_parquet_in_batches(input_path, batch_size=10000):
    """
    Read parquet file in batches to manage memory usage.

    Args:
        input_path (str): Path to parquet file
        batch_size (int): Number of rows to read at once

    Yields:
        pd.DataFrame: Batch of rows
    """
    try:
        # Read entire file (pandas doesn't support batch reading parquet directly)
        # For very large files, consider using pyarrow.dataset for true batched reading
        df = pd.read_parquet(input_path)
        total_rows = len(df)

        # Yield in batches
        for offset in range(0, total_rows, batch_size):
            yield df.iloc[offset:offset + batch_size], total_rows
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        sys.exit(1)


def convert_stories_to_text(input_path, output_path, num_stories=None,
                           remove_special_tokens=False, separator='\n',
                           batch_size=10000):
    """
    Convert stories from parquet to text file.

    Args:
        input_path (str): Path to input parquet file
        output_path (str): Path to output text file
        num_stories (int, optional): Number of stories to convert. If None, convert all.
        remove_special_tokens (bool): Whether to remove <START> and <STORY> tokens
        separator (str): Separator between stories
        batch_size (int): Batch size for processing
    """
    # Check if input file exists
    if not Path(input_path).exists():
        print(f"Error: Input file {input_path} does not exist")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    stories_written = 0

    print(f"Converting stories from {input_path} to {output_path}")
    print(f"Batch size: {batch_size}")
    print(f"Removing special tokens: {remove_special_tokens}")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Create progress bar
            pbar = None

            for batch_df, total_rows in read_parquet_in_batches(input_path, batch_size):
                if pbar is None:
                    # Initialize progress bar on first batch
                    pbar = tqdm.tqdm(total=total_rows if num_stories is None else min(num_stories, total_rows),
                                    desc="Converting stories")

                # Process each story in the batch
                for _, row in batch_df.iterrows():
                    if num_stories is not None and stories_written >= num_stories:
                        break

                    story = row['text']

                    # Remove special tokens if requested
                    if remove_special_tokens:
                        story = story.replace('<START>', '').replace('<STORY>', '').strip()

                    # Write story to file
                    f.write(story)

                    # Add separator (except for last story)
                    if (num_stories is None and stories_written < total_rows - 1) or \
                       (num_stories is not None and stories_written < num_stories - 1):
                        f.write(separator)

                    stories_written += 1

                total_processed += len(batch_df)
                pbar.update(min(len(batch_df), num_stories - stories_written if num_stories else len(batch_df)))

                # Break if we've reached the desired number of stories
                if num_stories is not None and stories_written >= num_stories:
                    break

            if pbar:
                pbar.close()

        print(f"\n✅ Conversion complete!")
        print(f"Total stories written: {stories_written}")
        print(f"Output file: {output_path}")
        print(f"File size: {Path(output_path).stat().st_size / (1024**2):.2f} MB")

    except Exception as e:
        print(f"Error during conversion: {e}")
        # Clean up partial output file
        if Path(output_path).exists():
            Path(output_path).unlink()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert TinyStories parquet file to text format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all stories to text
  python convert_to_txt.py --input data/tinystoriesclean.parquet --output data/tinystories.txt

  # Convert only 100,000 stories
  python convert_to_txt.py --input data/tinystoriesclean.parquet --output data/tinystories_100k.txt --num_stories 100000

  # Convert stories and remove special tokens
  python convert_to_txt.py --input data/tinystoriesclean.parquet --output data/tinystories_clean.txt --remove_special_tokens

  # Convert with custom separator
  python convert_to_txt.py --input data/tinystoriesclean.parquet --output data/tinystories_sep.txt --separator '===\n'
        """
    )

    parser.add_argument('--input', '-i', required=True,
                       help='Input parquet file path')
    parser.add_argument('--output', '-o', required=True,
                       help='Output text file path')
    parser.add_argument('--num_stories', '-n', type=int,
                       help='Number of stories to convert (default: all stories)')
    parser.add_argument('--remove_special_tokens', action='store_true',
                       help='Remove <START> and <STORY> tokens from output')
    parser.add_argument('--separator', '-s', default='\n',
                       help='Separator between stories (default: newline)')
    parser.add_argument('--batch_size', '-b', type=int, default=10000,
                       help='Batch size for processing (default: 10000)')

    args = parser.parse_args()

    # Validate arguments
    if args.num_stories is not None and args.num_stories <= 0:
        print("Error: num_stories must be a positive integer")
        sys.exit(1)

    if args.batch_size <= 0:
        print("Error: batch_size must be a positive integer")
        sys.exit(1)

    # Convert the data
    convert_stories_to_text(
        input_path=args.input,
        output_path=args.output,
        num_stories=args.num_stories,
        remove_special_tokens=args.remove_special_tokens,
        separator=args.separator,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()