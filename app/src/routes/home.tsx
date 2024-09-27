//import React, { useState, useEffect, ChangeEvent } from "react";

import UploadFile from "../components/uploadFile";
import Sidebar from "../components/sideBar";

export default function FileUploader() {
    
    return (
        <div className="bg-white min-h-screen flex min-w-[500px] min-h-[500px]">
            {/* Barra lateral */}
            <Sidebar />

            {/* Contenido principal */}
            <div className="flex-grow flex justify-center items-center">
                <UploadFile />
            </div>
        </div>
    );
}
