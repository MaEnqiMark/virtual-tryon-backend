import React, { useState, useRef } from "react";
import Webcam from "react-webcam";
import { TabContext, TabPanel, TabList } from '@mui/lab';
import { Box, Tab } from '@mui/material';
import ImageUpload from "./ImageUpload";

const Live = () => {
    const webcamRef = useRef(null);
    const [tabValue, setTabValue] = useState('upload')
    const [image, setImage] = useState(null)

    return (
        <div
            style={{
                display: 'flex',
                flexDirection: 'column'
            }}
        >
            <TabContext value={tabValue}>
                <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                    <TabList 
                        onChange={(e, val) => {
                            setTabValue(val)
                            setImage(null)
                        }} 
                    >
                    <Tab label="Upload Image" value="upload" />
                    <Tab label="Live Camera" value="live" />
                    </TabList>
                </Box>
                <TabPanel value="live">
                    <div
                        style={{
                            display: image ? 'none' : 'block',
                            height: '100%',
                            width: '100%'
                        }}
                    >
                        <Webcam
                            ref={webcamRef}
                            screenshotFormat="image/jpeg"
                            mirrored={true}
                            style={{
                                height: '100%',
                                width: '100%'
                            }}
                        />
                        <button
                            onClick={() => setImage(webcamRef.current.getScreenshot())}
                        >
                            Capture photo
                        </button>
                    </div>

                    {image && (
                        <div>
                            <img
                                src={image}
                            />

                            <button
                                onClick={() => setImage(null)}
                            >
                                Retake photo
                            </button>
                        </div>
                    )}
                </TabPanel>
                <TabPanel value="upload">
                    <ImageUpload
                        handleUpload={img => {
                            setImage(img)
                        }}
                    />
                </TabPanel>
            </TabContext>
        </div>
    )
}

export default Live