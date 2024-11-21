# Create or empty the final output file
> ikat_2023_passage_index.tar.bz2

echo "created empty file. starting process..."

# Loop through each part and append to the final file
for part in ikat_2023_passage_index.tar.bz2.part*; do
  cat "$part" >> ikat_2023_passage_index.tar.bz2
  echo "done $part"
done

echo "done"
