#!/bin/zsh

echo "Renaming textures..."

typeset -A RENAME_MAP
RENAME_MAP=(
  AmbientOcclusion ambient-occlusion
  Color albedo
  Displacement displacement
  NormalGL normal
  Roughness roughness
)

for DIR in ./*(/); do
  echo "Processing directory: $DIR"

  for OLD_KEY in "${(@k)RENAME_MAP}"; do    
    FILES=($DIR/*$OLD_KEY.*(N))

    if (( ${#FILES} == 0 )); then
      continue
    fi

    for FILE in $FILES; do
      BASENAME="${DIR:t}"
      NEW_NAME="${BASENAME}_${RENAME_MAP[$OLD_KEY]}"
      EXT="${FILE##*.}"
      NEW_PATH="$DIR/$NEW_NAME.$EXT"

      echo "→ Renaming: $FILE"
      echo "→ To:       $NEW_PATH"
      mv "$FILE" "$NEW_PATH"
    done
  done
done

echo "Done."