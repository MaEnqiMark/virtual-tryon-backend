import { Button } from "@mui/material";
import { useState } from "react";

const Slideshow = ({ images = [] }) => {
  const [index, setIndex] = useState(0);

  const nextSlide = () => {
    setIndex((prev) => (prev + 1) % images.length);
  };

  const prevSlide = () => {
    setIndex((prev) => (prev - 1 + images.length) % images.length);
  };

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        overflow: "hidden",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <img
        src={images[index]}
        style={{
          maxWidth: "100%",
          maxHeight: "100%",
          objectFit: "contain",
          display: "block",
        }}
      />

      <Button
        onClick={prevSlide}
        style={{
          position: "absolute",
          left: "10px",
          top: "50%",
          transform: "translateY(-50%)",
          minWidth: "20px",
          height: "60px",
          padding: 0,
          fontSize: "18px",
          background: "rgba(0,0,0,0.5)",
          color: "white",
        }}
      >
        {"<"}
      </Button>

      <Button
        onClick={nextSlide}
        style={{
          position: "absolute",
          right: "10px",
          top: "50%",
          transform: "translateY(-50%)",
          minWidth: "20px",
          height: "60px",
          padding: 0,
          fontSize: "18px",
          background: "rgba(0,0,0,0.5)",
          color: "white",
        }}
      >
        {">"}
      </Button>
    </div>
  );
};

export default Slideshow;
