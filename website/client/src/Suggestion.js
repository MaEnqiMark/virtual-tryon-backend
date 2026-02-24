import { Button, TextField, Menu, MenuItem } from "@mui/material"
import { useEffect, useState } from "react"
import Cloth from "./Cloth"
import Slideshow from "./Slideshow"
import ImageUpload from "./ImageUpload"

const clothing_url = 'https://virtual-tryon-backend-974u.onrender.com'

const Suggestion = () => {
    
    const [metadata, setMetadata] = useState([])
    const [mode, setMode] = useState('tops')
    const [cloth, setCloth] = useState(null)
    const [query, setQuery] = useState('')
    const [result, setResult] = useState(null)

    useEffect(() => {
        if (mode != 'upload') {
            fetch(`${clothing_url}/${mode}`)
                .then(res => res.json())
                .then(json => {
                    json = json.map((item) => item.image_url)
                    const result = []
                    for (let i = 0; i < 9; i++) {
                        const randomIndex = Math.floor(Math.random() * json.length);
                        const selectedElement = clothing_url + json.splice(randomIndex, 1)[0]; // Remove and get the element
                        result.push({'img': selectedElement});
                    }
                    setMetadata(result)
                })
        } else { // mode = 'upload'
            setMetadata(null)
        }
    }, [mode])

    const getSuggestion = (event) => {
        // get suggestion using query + selected clothing
        if (cloth != null) {
            if (mode == 'tops') {
                const data = {
                    "top_id": cloth
                }
                fetch(`${clothing_url}/generate_looks_from_top`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                })
                    .then(res => res.json())
                    .then(json => {
                        console.log(json)
                        setResult(json)
                    })
            } else if (mode == 'tops') {
                const data = {
                    "top_id": cloth
                }
                fetch(`${clothing_url}/generate_looks_from_top`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                })
                    .then(res => res.json())
                    .then(json => {
                        console.log(json)
                        setResult(json)
                    })
            } else if (mode == 'bottoms') {
                const data = {
                    "bottom_id": cloth
                }
                fetch(`${clothing_url}/generate_looks_from_bottom`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                })
                    .then(res => res.json())
                    .then(json => {
                        console.log(json)
                        setResult(json)
                    })
            } else {
                const file = new File([cloth], 'image.jpg', {type: cloth.type})
                const formData = new FormData()
                formData.append("file", file)
                fetch(`${clothing_url}/suggest_from_photo`, {
                    method: 'POST',
                    body: formData
                })
                    .then(res => res.json())
                    .then(json => {
                        console.log(json)
                        setResult(json)
                    })

            }
        }
    }

    const [anchorEl, setAnchorEl] = useState(null);
    const open = Boolean(anchorEl);
    const handleClick = (event) => {
        setAnchorEl(event.currentTarget);
    };
    const handleClose = (mode) => {
        setMode(mode)
        setAnchorEl(null);
    };



    return (
        <div
            style={{
                display: 'flex',
                flexDirection: 'row'
            }}
        >
            <div
                style={{
                    height: '100%',
                    width: '100%'
                }}
            >
                <div>
                    <Button
                        id="basic-button"
                        aria-controls={open ? 'basic-menu' : undefined}
                        aria-haspopup="true"
                        aria-expanded={open ? 'true' : undefined}
                        onClick={handleClick}
                    >
                        Options
                    </Button>
                    <Menu
                        id="basic-menu"
                        anchorEl={anchorEl}
                        open={open}
                        onClose={handleClose}
                        slotProps={{
                            list: {
                                'aria-labelledby': 'basic-button',
                            },
                        }}
                    >
                        <MenuItem onClick={() => handleClose('tops')}>Top</MenuItem>
                        <MenuItem onClick={() => handleClose('bottoms')}>Bottom</MenuItem>
                        <MenuItem onClick={() => handleClose('upload')}>Upload</MenuItem>
                    </Menu>
                </div>

                <div
                    style={{
                        height: '100%',
                        width: '100%'
                    }}
                >
                    {metadata ? (
                        <Cloth
                            id='suggestion'
                            itemData={metadata}
                            num_cols={3}
                            handleSelection={(img) => {
                                img = img.substring((clothing_url+'/static/').length, img.length-4)
                                setCloth(img)
                            }}
                            upload={false}
                        />
                    ) : (
                        <div>
                            <ImageUpload
                                handleUpload={(img) => setCloth(img)}
                            />
                        </div>
                    )}
                </div>
            </div>

            <div
                style={{
                    display: 'flex',
                    flexDirection: 'column',
                    height: '100%',
                    width: '100%'
                }}
            >
                {/* {mode == 'upload' && 
                    <TextField
                        label='enter query'
                        variant='standard'
                        multiline
                        onChange={(e) => {
                            setQuery(e.target.value)
                        }}
                    />
                } */}

                <Button
                    onClick={getSuggestion}
                >
                    Get suggestion
                </Button>

                {result && (
                    <div
                        style={{
                            display: 'flex',
                            flexDirection: 'column',
                            maxHeight: '100vh',
                            overflowX: 'scroll'
                        }}
                    >
                        Explanation: {result.explanation}
                        <Slideshow images={result.looks.map((item) => `https://virtual-tryon-backend-974u.onrender.com${item.bottom.image_url}`)}/>
                        {/* {result.looks.map((item) => (
                            <img 
                                key={item.bottom.image_url} 
                                src={`https://virtual-tryon-backend-974u.onrender.com${item.bottom.image_url}`}
                                style={{
                                    width: '100%',
                                    height: '100%'
                                }}
                            />
                        ))} */}
                    </div>
                )}
            </div>
        </div>
    )

}

export default Suggestion