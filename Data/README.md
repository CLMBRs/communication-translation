
# Prepare data for backtranslation

1. Download data from [CC-100](https://data.statmt.org/cc-100/) website
2. Run `head -500000 <language>.txt > <language>_500K.txt`
3. (Optional) For random sampling from `<language>.txt`, consider using `shuf <language>.txt > | head -500000 > <language>_500K.txt`. If the file is too large to fit in memory, consider using [terashuf](https://github.com/alexandres/terashuf)
