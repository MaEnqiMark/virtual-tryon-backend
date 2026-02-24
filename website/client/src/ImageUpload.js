import { useRef, useState } from "react";

const ImageUpload = ({handleUpload}) => {
	const inputRef = useRef(null)
	const [image, setImage] = useState(null)
	
	const handlePaste = async (event) => {
		const items = event.clipboardData.items;
		for (let i = 0; i < items.length; i++) {
			const item = items[i]
			if (item.type.includes('image/')) {
				const blob = await item.getAsFile()
				const url = URL.createObjectURL(blob)
				setImage(url)
				handleUpload(url)
				break
			}
		}
	}
	
	const handleInput = (event) => {
		const file = event.target.files[0]
		const url = URL.createObjectURL(file)
		setImage(url)
		handleUpload(url)
		event.target.value = ""
	}
	
	return (
		
		<div
			style={{
				height: '100%',
				width: '100%',
				boxSizing: 'border-box',
				
				cursor: 'pointer',
				border: '2px dashed #ccc',

				overflow: 'hidden',
				
				display: 'flex',
				alignItems: 'center',
				justifyContent: 'center'
			}}
			onPaste={handlePaste}
			onClick={e => inputRef.current.click()}
		>
		
			<input
				ref={inputRef}
				type='file'
				onChange={handleInput}
				style={{
					display: 'none'
				}}
			/>
			
			
			{image ? (
				<div
					style={{
						position: "relative",
						display: "inline-block",
					}}
					
				>
					<img 
						src={image}
						style={{
              maxWidth: "100%",
              maxHeight: "100%",
              width: "100%",
              height: "100%",
              objectFit: "contain",
              display: "block",
            }}
					/>
					<button
						onClick={(e) => {
							e.stopPropagation();
							setImage(null)
						}}
						style={{
							position: "absolute",
							top: 5,
							right: 5,
							background: "black",
							color: "white",
							border: "none",
							borderRadius: "20%",
							width: 24,
							height: 24,
							cursor: "pointer",
							fontSize: 14,
							lineHeight: "24px",
							textAlign: "center"
						}}
						>
						x
					</button>
				</div>
			) : (
				<p>Upload image or paste image</p>
			)}
		</div>
	)
}

export default ImageUpload