export default function FileUploader() {
    return (
        <div className="bg-white min-h-screen flex flex-col items-center justify-center p-4">
            <h1 className="text-2xl font-bold mb-4">Model Accuracy: 89.00%</h1>

            <div className="flex flex-wrap justify-center gap-4">
                {/* Imagen 1 */}
                <img 
                    src="../../public/imgs/confusion_matrix.png" 
                    alt="Imagen 1"
                    className="w-full sm:w-1/2 lg:w-1/3 object-contain"
                />
                {/* Imagen 2 */}
                <img 
                    src="../../public/imgs/loss_graph.png" 
                    alt="Imagen 2"
                    className="w-full sm:w-1/2 lg:w-1/3 object-contain"
                />
            </div>
        </div>
    );
}