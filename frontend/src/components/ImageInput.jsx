import { useState } from "react";

export default function ImageInput({ onFileSelected }) {
  const [preview, setPreview] = useState(null);

  function handleFile(file) {
    if (!file) return;
    onFileSelected(file);
    setPreview(URL.createObjectURL(file));
  }

  return (
    <div className="image-input">
      <label className="dropzone">
        <input
          type="file"
          accept="image/*"
          onChange={(e) => handleFile(e.target.files[0])}
        />
        {preview ? (
          <img src={preview} alt="Selected upload preview" className="image-preview" />
        ) : (
          <span>Click or drop an image</span>
        )}
      </label>
    </div>
  );
}
