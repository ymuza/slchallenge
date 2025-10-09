from argparse import ArgumentParser
import h5py
import numpy as np
import torch
from tqdm import tqdm

from astroclip.data.datamodule import AstroClipCollator, AstroClipDataloader
from astroclip.models.astroclip import AstroClipModel


def embed_astroclip(
    model_path: str,
    dataset_path: str,
    save_path: str,
    max_size: int = None,
    batch_size: int = 256,
    loader_type: str = "val",
):

    print("Loading the models")
    """Extract embeddings from the AstroClip model and save them incrementally to an HDF5 file"""
    # Load the model
    astroclip = AstroClipModel.load_from_checkpoint(model_path)

    # Get the dataloader
    loader = AstroClipDataloader(
        path=dataset_path,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=AstroClipCollator(),
        columns=["image", "spectrum", "targetid"],
    )
    loader.setup("fit")

    # Set up loader
    if loader_type == "train":
        loader = loader.train_dataloader()
    elif loader_type == "val":
        loader = loader.val_dataloader()
    else:
        raise ValueError("loader must be either 'train' or 'val'")

    print("Loading the data")
    
    # Prepare HDF5 file with extendable astro_datasets
    with h5py.File(save_path, "w") as f:
        total_samples = 0
        initialized = False

        # Process batches
        with torch.no_grad():
            for idx, batch_test in tqdm(enumerate(loader), desc="Extracting embeddings"):
                if max_size is not None and total_samples >= max_size:
                    break

                # Compute embeddings
                im_embed_batch = astroclip(batch_test["image"].cuda(), input_type="image").cpu().detach().numpy()
                sp_embed_batch = astroclip(batch_test["spectrum"].cuda(), input_type="spectrum").cpu().detach().numpy()
                obj_ids_batch = batch_test["targetid"].numpy()
                images_batch = batch_test["image"].cpu().numpy()
                spectra_batch = batch_test["spectrum"].cpu().numpy()

                # Initialize astro_datasets on the first batch
                if not initialized:
                    image_embed_shape = (0, im_embed_batch.shape[1])
                    spectrum_embed_shape = (0, sp_embed_batch.shape[1])
                    image_shape = (0, *images_batch.shape[1:])
                    spectrum_shape = (0, *spectra_batch.shape[1:])
                    obj_id_shape = (0,)

                    image_embeddings = f.create_dataset("image_embeddings", shape=image_embed_shape, maxshape=(None, im_embed_batch.shape[1]), dtype="float32")
                    spectrum_embeddings = f.create_dataset("spectrum_embeddings", shape=spectrum_embed_shape, maxshape=(None, sp_embed_batch.shape[1]), dtype="float32")
                    images = f.create_dataset("image", shape=image_shape, maxshape=(None, *image_shape[1:]), dtype="float32")
                    spectra = f.create_dataset("spectrum", shape=spectrum_shape, maxshape=(None, *spectrum_shape[1:]), dtype="float32")
                    object_ids = f.create_dataset("object_id", shape=obj_id_shape, maxshape=(None,), dtype="int64")

                    initialized = True

                # Determine batch size
                batch_size_actual = im_embed_batch.shape[0]

                # Resize astro_datasets to accommodate new data
                image_embeddings.resize(total_samples + batch_size_actual, axis=0)
                spectrum_embeddings.resize(total_samples + batch_size_actual, axis=0)
                images.resize(total_samples + batch_size_actual, axis=0)
                spectra.resize(total_samples + batch_size_actual, axis=0)
                object_ids.resize(total_samples + batch_size_actual, axis=0)

                # Write new data
                image_embeddings[total_samples: total_samples + batch_size_actual] = im_embed_batch
                spectrum_embeddings[total_samples: total_samples + batch_size_actual] = sp_embed_batch
                images[total_samples: total_samples + batch_size_actual] = images_batch
                spectra[total_samples: total_samples + batch_size_actual] = spectra_batch
                object_ids[total_samples: total_samples + batch_size_actual] = obj_ids_batch

                # Update sample count
                total_samples += batch_size_actual

                # Clear memory
                del batch_test, im_embed_batch, sp_embed_batch, images_batch, spectra_batch
                torch.cuda.empty_cache()

    print(f"Embeddings saved to {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "save_path",
        type=str,
        help="Path to save the embeddings",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
        default=256,
    )
    parser.add_argument(
        "--max_size",
        type=int,
        help="Maximum number of samples to use",
        default=None,
    )
    parser.add_argument(
        "--loader_type",
        type=str,
        help="Which loader to use (train or val)",
        default="val",
    )

    print("Embedding ascroclip")
    args = parser.parse_args()
    embed_astroclip(
        args.model_path,
        args.dataset_path,
        args.save_path,
        args.max_size,
        args.batch_size,
        args.loader_type,
    )
