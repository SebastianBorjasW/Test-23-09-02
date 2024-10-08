//import React, { useState, useEffect, ChangeEvent } from "react";

import UploadFile from "../components/uploadFile";
import Sidebar from "../components/sideBar";

export default function FileUploader() {
    
    return (
        <div className="bg-white min-h-screen flex">
            {/* Barra lateral */}
            <Sidebar />

            {/* Contenido principal */}
            <div className="flex-grow flex justify-center items-center">
                <UploadFile />
            </div>
        </div>
    );
}
