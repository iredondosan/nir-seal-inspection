#!/bin/bash
# Rename prod6 -> prod6 (and prod6_bad -> prod6_bad) across data + code. Uniform substring replace.
set -e
cd /home/ubuntu/TFM/seal-inspection
echo "backup annotations -> /tmp/annotations_backup_prod6"
rm -rf /tmp/annotations_backup_prod6 && cp -r data/annotations /tmp/annotations_backup_prod6

echo "1) rename image files inside the two dirs"
for d in prod6 prod6_bad; do
  if [ -d "data/images/$d" ]; then
    (cd "data/images/$d" && for f in prod6_*; do [ -e "$f" ] && mv -- "$f" "${f/prod6/prod6}"; done)
  fi
done
echo "2) rename the image dirs"
[ -d data/images/prod6 ]     && mv data/images/prod6 data/images/prod6
[ -d data/images/prod6_bad ] && mv data/images/prod6_bad data/images/prod6_bad

echo "3) rewrite + rename annotation XMLs (content name= attrs and filenames)"
for x in data/annotations/prod6*.xml; do [ -e "$x" ] && sed -i 's/prod6/prod6/g' "$x"; done
(cd data/annotations && for x in prod6*.xml; do [ -e "$x" ] && mv -- "$x" "${x/prod6/prod6}"; done)

echo "4) rewrite code references (src/ + seal_inspection/)"
for f in $(grep -rl 'prod6' --include='*.py' src seal_inspection 2>/dev/null); do sed -i 's/prod6/prod6/g' "$f"; done

echo "=== VERIFY ==="
echo "image dirs:"; ls -d data/images/prod6 data/images/prod6_bad 2>&1
echo "prod6 files sample:"; ls data/images/prod6 | head -2; ls data/images/prod6_bad | head -2
echo "annotations:"; ls data/annotations/ | grep -E 'prod6|prod6'
echo "any prod6 left in code?"; grep -rl 'prod6' --include='*.py' src seal_inspection 2>/dev/null || echo '  none'
echo "any prod6 left in annotations?"; grep -l 'prod6' data/annotations/*.xml 2>/dev/null || echo '  none'
echo "DONE"
