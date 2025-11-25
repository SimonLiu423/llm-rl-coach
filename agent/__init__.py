from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from . import agent

__all__ = ["agent"]

# # Configure the tracer provider and add the exporter
# tracer_provider = TracerProvider()
# tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
# trace.set_tracer_provider(tracer_provider)
