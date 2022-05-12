mod protos;
mod renderer;
mod svg_writer;

use protos::service::psvg_server::{Psvg, PsvgServer};
use protos::service::{
    ConvertReply, ConvertRequest, RenderReply, RenderRequest,
};
use tonic::{transport::Server, Request, Response, Status};

#[derive(Debug, Default)]
pub struct PsvgServiceImpl {}

#[tonic::async_trait]
impl Psvg for PsvgServiceImpl {
    async fn convert(
        &self,
        request: Request<ConvertRequest>,
    ) -> Result<Response<ConvertReply>, Status> {
        let svg = request.into_inner().svg;

        if let Some(proto_svg) = svg {
            let svg_xml = svg_writer::protosvg_to_svg(proto_svg)
                .map_err(Status::invalid_argument)?;

            Ok(Response::new(ConvertReply { svg_xml }))
        } else {
            Err(Status::invalid_argument("No ProtoSVG received"))
        }
    }

    async fn render(
        &self,
        request: Request<RenderRequest>,
    ) -> Result<Response<RenderReply>, Status> {
        let svg = request.into_inner().svg;

        if let Some(proto_svg) = svg {
            let svg_xml = svg_writer::protosvg_to_svg(proto_svg)
                .map_err(Status::invalid_argument)?;

            let pixmap =
                renderer::render(&svg_xml).map_err(Status::internal)?;

            let png_data = pixmap
                .encode_png()
                .map_err(|e| Status::internal(e.to_string()))?;

            Ok(Response::new(RenderReply { svg_xml, png_data }))
        } else {
            Err(Status::invalid_argument("No ProtoSVG received"))
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::0]:50051".parse()?;
    let service = PsvgServiceImpl::default();

    Server::builder()
        .add_service(PsvgServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
